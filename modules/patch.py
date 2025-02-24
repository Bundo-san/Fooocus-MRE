import torch
import contextlib
import comfy.model_base
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.samplers
import comfy.model_management
import modules.anisotropic as anisotropic
import comfy.ldm.modules.attention
import comfy.k_diffusion.sampling
import comfy.sd1_clip
import modules.inpaint_worker as inpaint_worker
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.ldm.modules.diffusionmodules.model
import comfy.sd

from comfy.model_base import sdxl_pooled
from comfy.k_diffusion import utils
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler, trange
from comfy.ldm.modules.diffusionmodules.openaimodel import timestep_embedding, forward_timestep_embed


sharpness = 2.0
negative_adm = True

comfy_encode_adm = comfy.model_base.SDXL.encode_adm

cfg_x0 = 0.0
cfg_s = 1.0
cfg_cin = 1.0


def calculate_weight_patched(self, patches, weight, key):
    for p in patches:
        alpha = p[0]
        v = p[1]
        strength_model = p[2]

        if strength_model != 1.0:
            weight *= strength_model

        if isinstance(v, list):
            v = (self.calculate_weight(v[1:], v[0].clone(), key), )

        if len(v) == 1:                                                                                                                                                                                          
            patch_type = "diff"                                                           
        elif len(v) == 2:                                                                 
            patch_type = v[0]                                                             
            v = v[1]
        elif len(v) == 3:
            patch_type = "fooocus"
                                                                                              
        if patch_type == "diff":
            # Code from ComfyUI
            w1 = v[0]
            if alpha != 0.0:
                if w1.shape != weight.shape:
                    print("WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}"
                          .format(key, w1.shape, weight.shape))
                else:
                    weight += alpha * comfy.model_management.cast_to_device(w1, weight.device,
                                                                            weight.dtype)
        elif patch_type == "fooocus":
            # fooocus
            w1 = comfy.model_management.cast_to_device(v[0], weight.device, torch.float32)
            w_min = comfy.model_management.cast_to_device(v[1], weight.device, torch.float32)
            w_max = comfy.model_management.cast_to_device(v[2], weight.device, torch.float32)
            w1 = (w1 / 255.0) * (w_max - w_min) + w_min
            if alpha != 0.0:
                if w1.shape != weight.shape:
                    print("WARNING SHAPE MISMATCH {} FOOOCUS WEIGHT NOT MERGED {} != {}"
                          .format(key, w1.shape, weight.shape))
                else:
                    weight += alpha * comfy.model_management.cast_to_device(w1, weight.device,
                                                                            weight.dtype)
        elif patch_type == "lora": #lora/locon
            # Code from ComfyUI
            mat1 = comfy.model_management.cast_to_device(v[0], weight.device, torch.float32)
            mat2 = comfy.model_management.cast_to_device(v[1], weight.device, torch.float32)
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            if v[3] is not None:
                #locon mid weights, hopefully the math is fine because I didn't properly test it
                mat3 = comfy.model_management.cast_to_device(v[3], weight.device, torch.float32)
                final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1)
                                .flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
            try:
                weight += (alpha * torch.mm(mat1.flatten(start_dim=1), 
                            mat2.flatten(start_dim=1))).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                print("ERROR", key, e)
        elif patch_type == "lokr": #lokr
            w1 = v[0]
            w2 = v[1]
            w1_a = v[3]
            w1_b = v[4]
            w2_a = v[5]
            w2_b = v[6]
            t2 = v[7]
            dim = None

            if w1 is None:
                dim = w1_b.shape[0]
                w1 = torch.mm(comfy.model_management.cast_to_device(w1_a, weight.device, torch.float32),
                              comfy.model_management.cast_to_device(w1_b, weight.device, torch.float32))
            else:
                w1 = comfy.model_management.cast_to_device(w1, weight.device, torch.float32)

            if w2 is None:
                dim = w2_b.shape[0]
                if t2 is None:
                    w2 = torch.mm(comfy.model_management
                                  .cast_to_device(w2_a, weight.device, torch.float32),
                                  comfy.model_management
                                  .cast_to_device(w2_b, weight.device, torch.float32))
                else:
                    w2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                      comfy.model_management
                                      .cast_to_device(t2, weight.device, torch.float32),
                                      comfy.model_management
                                      .cast_to_device(w2_b, weight.device, torch.float32),
                                      comfy.model_management
                                      .cast_to_device(w2_a, weight.device, torch.float32))
            else:
                w2 = comfy.model_management.cast_to_device(w2, weight.device, torch.float32)

            if len(w2.shape) == 4:
                w1 = w1.unsqueeze(2).unsqueeze(2)
            if v[2] is not None and dim is not None:
                alpha *= v[2] / dim

            try:
                weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                print("ERROR", key, e)
        elif patch_type == "loha":
            w1a = v[0]
            w1b = v[1]
            if v[2] is not None:
                alpha *= v[2] / w1b.shape[0]
            w2a = v[3]
            w2b = v[4]
            if v[5] is not None: #cp decomposition
                t1 = v[5]
                t2 = v[6]
                m1 = torch.einsum('i j k l, j r, i p -> p r k l',
                                  comfy.model_management
                                  .cast_to_device(t1, weight.device, torch.float32),
                                  comfy.model_management
                                  .cast_to_device(w1b, weight.device, torch.float32),
                                  comfy.model_management
                                  .cast_to_device(w1a, weight.device, torch.float32))

                m2 = torch.einsum('i j k l, j r, i p -> p r k l',
                                  comfy.model_management
                                  .cast_to_device(t2, weight.device, torch.float32),
                                  comfy.model_management
                                  .cast_to_device(w2b, weight.device, torch.float32),
                                  comfy.model_management
                                  .cast_to_device(w2a, weight.device, torch.float32))
            else:
                m1 = torch.mm(comfy.model_management.cast_to_device(w1a, weight.device, torch.float32),
                              comfy.model_management.cast_to_device(w1b, weight.device, torch.float32))
                m2 = torch.mm(comfy.model_management.cast_to_device(w2a, weight.device, torch.float32),
                              comfy.model_management.cast_to_device(w2b, weight.device, torch.float32))

            try:
                weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                print("ERROR", key, e)
        elif patch_type == "glora":
            if v[4] is not None:
                alpha *= v[4] / v[0].shape[0]

            a1 = comfy.model_management.cast_to_device(v[0].flatten(start_dim=1), 
                                                       weight.device, torch.float32)                                                                                                  
            a2 = comfy.model_management.cast_to_device(v[1].flatten(start_dim=1),
                                                       weight.device, torch.float32)
            b1 = comfy.model_management.cast_to_device(v[2].flatten(start_dim=1),
                                                       weight.device, torch.float32)
            b2 = comfy.model_management.cast_to_device(v[3].flatten(start_dim=1),
                                                       weight.device, torch.float32)
                                                                                              
            weight += ((torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1),
                            a2), a1)) * alpha).reshape(weight.shape).type(weight.dtype)
        else:                                                                             
            print("patch type not recognized", patch_type, key)

    return weight

# Modify the CFG values by applying sharpness to the image.
# Called by sampling_function() in comfy/samplers.py.
def cfg_patched(args):
    global cfg_x0, cfg_s
    positive_eps = args['cond'].clone()
    positive_x0 = args['cond'] * cfg_s + cfg_x0
    uncond = args['uncond'] * cfg_s + cfg_x0
    cond_scale = args['cond_scale']
    t = args['timestep']

    alpha = 1.0 - (t / 999.0)[:, None, None, None].clone()
    alpha *= 0.001 * sharpness

    eps_degraded = anisotropic.adaptive_anisotropic_filter(x=positive_eps, g=positive_x0)
    eps_degraded_weighted = eps_degraded * alpha + positive_eps * (1.0 - alpha)

    cond = eps_degraded_weighted * cfg_s + cfg_x0

    return uncond + (cond - uncond) * cond_scale


def text_encoder_device_patched():
    # Fooocus's style system uses text encoder much more times than comfy so this makes things much faster.
    return comfy.model_management.get_torch_device()


@torch.no_grad()
def sample_dpmpp_fooocus_2m_sde_inpaint_seamless(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, **kwargs):
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    assert isinstance(seed, int)
    old_denoised = None
    h_last = None
    h = None

    #========================================================================================
    # Fooocus-Exclusive Code Starts Here:
    #========================================================================================
    # Generate different seed with CPU:
    energy_generator = torch.Generator(device='cpu')
    energy_generator.manual_seed(seed + 1)  # avoid bad results by using different seeds.

    def get_energy():
        return torch.randn(x.size(), dtype=x.dtype, generator=energy_generator, device="cpu").to(x)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler

    # Note: model.inner_model is CFGNoisePredictor object.
    latent_processor = model.inner_model.inner_model.process_latent_in
    inpaint_latent = None
    inpaint_mask = None

    if inpaint_worker.current_task is not None:
        inpaint_latent = latent_processor(inpaint_worker.current_task.latent).to(x)
        inpaint_mask = inpaint_worker.current_task.latent_mask.to(x)

    def blend_latent(a, b, w):
        return a * w + b * (1 - w)
    #========================================================================================
    # Fooocus-Exclusive Code Ends
    #========================================================================================

    for i in trange(len(sigmas) - 1, disable=disable):
        if inpaint_latent is None:
            denoised = model(x, sigmas[i] * s_in, **extra_args)
        else:
            energy = get_energy() * sigmas[i] + inpaint_latent
            x_prime = blend_latent(x, energy, inpaint_mask)
            denoised = model(x_prime, sigmas[i] * s_in, **extra_args)
            denoised = blend_latent(denoised, inpaint_latent, inpaint_mask)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised
            if old_denoised is not None:
                r = h_last / h
                x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h

    return x


def patched_unet_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """
    inpaint_fix = None
    if inpaint_worker.current_task is not None:
        inpaint_fix = inpaint_worker.current_task.inpaint_head_feature

    transformer_options["original_shape"] = list(x.shape)
    transformer_options["current_index"] = 0
    transformer_patches = transformer_options.get("patches", {})

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for id, module in enumerate(self.input_blocks):
        transformer_options["block"] = ("input", id)
        h = forward_timestep_embed(module, h, emb, context, transformer_options)

        if inpaint_fix is not None:
            if int(h.shape[1]) == int(inpaint_fix.shape[1]):
                h = h + inpaint_fix.to(h)
                inpaint_fix = None

        if control is not None and 'input' in control and len(control['input']) > 0:
            ctrl = control['input'].pop()
            if ctrl is not None:
                h += ctrl
        hs.append(h)
    transformer_options["block"] = ("middle", 0)
    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
    if control is not None and 'middle' in control and len(control['middle']) > 0:
        ctrl = control['middle'].pop()
        if ctrl is not None:
            h += ctrl

    for id, module in enumerate(self.output_blocks):
        transformer_options["block"] = ("output", id)
        hsp = hs.pop()
        if control is not None and 'output' in control and len(control['output']) > 0:
            ctrl = control['output'].pop()
            if ctrl is not None:
                hsp += ctrl

        if "output_block_patch" in transformer_patches:
            patch = transformer_patches["output_block_patch"]
            for p in patch:
                h, hsp = p(h, hsp, transformer_options)

        h = torch.cat([h, hsp], dim=1)
        del hsp
        if len(hs) > 0:
            output_shape = hs[-1].shape
        else:
            output_shape = None
        h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)


def patched_SD1ClipModel_forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        if backup_embeds.weight.dtype != torch.float32:
            precision_scope = torch.autocast
        else:
            precision_scope = contextlib.nullcontext

        with precision_scope(comfy.model_management.get_autocast_device(device)):
            outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
            self.transformer.set_input_embeddings(backup_embeds)

            if self.layer == "last":
                z = outputs.last_hidden_state
            elif self.layer == "pooled":
                z = outputs.pooler_output[:, None, :]
            else:
                z = outputs.hidden_states[self.layer_idx]
                if self.layer_norm_hidden_state:
                    z = self.transformer.text_model.final_layer_norm(z)

            pooled_output = outputs.pooler_output
            if self.text_projection is not None:
                pooled_output = pooled_output.float().to(self.text_projection.device) @ self.text_projection.float()
        return z.float(), pooled_output.float()


VAE_DTYPE = None


def vae_dtype_patched():
    global VAE_DTYPE
    if VAE_DTYPE is None:
        VAE_DTYPE = torch.float32
        if comfy.model_management.is_nvidia():
            torch_version = torch.version.__version__
            if int(torch_version[0]) >= 2:
                if torch.cuda.is_bf16_supported():
                    VAE_DTYPE = torch.bfloat16
                    print('BFloat16 VAE: Enabled')
    return VAE_DTYPE


def vae_bf16_upsample_forward(self, x):
    try:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    except:  # operation not implemented for bf16
        b, c, h, w = x.shape
        out = torch.empty((b, c, h * 2, w * 2), dtype=x.dtype, layout=x.layout, device=x.device)
        split = 8
        l = out.shape[1] // split
        for i in range(0, out.shape[1], l):
            out[:, i:i + l] = torch.nn.functional.interpolate(x[:, i:i + l].to(torch.float32), scale_factor=2.0,
                                                              mode="nearest").to(x.dtype)
        del x
        x = out

    if self.with_conv:
        x = self.conv(x)
    return x


def patch_all():
    # Modify how CFG's are calculated to factor in sharpness:
    comfy.model_patcher.ModelPatcher.calculate_weight = calculate_weight_patched
    comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = patched_unet_forward

    # Add custom sampler to handle inpainting:
    comfy.k_diffusion.sampling.sample_dpmpp_fooocus_2m_sde_inpaint_seamless = sample_dpmpp_fooocus_2m_sde_inpaint_seamless

    comfy.model_management.text_encoder_device = text_encoder_device_patched
    print(f'Fooocus Text Processing Pipelines are retargeted to {str(comfy.model_management.text_encoder_device())}')

    return
