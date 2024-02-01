from comfy.samplers import *

import comfy.model_management
import modules.virtual_memory

KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",   
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", 
                  "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde",
                  "dpmpp_3m_sde_gpu", "ddpm", "lcm"]

#The main sampling function shared by all the samplers                                        
#Returns denoised                                                                             
def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None                                                                    
        else:                                                                                 
            uncond_ = uncond                                                                  
                                                                                              
        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)
        if "sampler_cfg_function" in model_options:                                           
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
            cfg_result = x - model_options["sampler_cfg_function"](args)                      
        else:                                                                                 
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale                 
                                                                                              
        for fn in model_options.get("sampler_post_cfg_function", []):                         
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}            
            cfg_result = fn(args)                                                             
                                                                                              
        return cfg_result

class CFGNoisePredictor(torch.nn.Module):                                                                                                                                                                            
    def __init__(self, model):                                                                
        super().__init__()                                                                    
        self.inner_model = model                                                              
    def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None):
        out = sampling_function(self.inner_model, x, timestep, uncond, cond, cond_scale, model_options=model_options, seed=seed)
        return out                                                                            
    def forward(self, *args, **kwargs):                                                       
        return self.apply_model(*args, **kwargs)                                              
                                                                                              
class KSamplerX0Inpaint(torch.nn.Module):                                                     
    def __init__(self, model):                                                                
        super().__init__()                                                                    
        self.inner_model = model                                                              
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:                                                          
            latent_mask = 1. - denoise_mask                                                   
            x = x * denoise_mask + (self.latent_image + self.noise * sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1))) * latent_mask
        out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, model_options=model_options, seed=seed)
        if denoise_mask is not None:                                                          
            out = out * denoise_mask + self.latent_image * latent_mask                        
        return out

def simple_scheduler(model, steps):                                                                                                                                                                                  
    s = model.model_sampling                                                                  
    sigs = []                                                                                 
    ss = len(s.sigmas) / steps                                                                
    for x in range(steps):                                                                    
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]                                         
    sigs += [0.0]                                                                             
    return torch.FloatTensor(sigs)                                                            
                                                                                              
def ddim_scheduler(model, steps):                                                             
    s = model.model_sampling                                                                  
    sigs = []                                                                                 
    ss = len(s.sigmas) // steps                                                               
    x = 1                                                                                     
    while x < len(s.sigmas):                                                                  
        sigs += [float(s.sigmas[x])]                                                          
        x += ss                                                                               
    sigs = sigs[::-1]                                                                         
    sigs += [0.0]                                                                             
    return torch.FloatTensor(sigs)                                                            
                                                                                              
def normal_scheduler(model, steps, sgm=False, floor=False):                                   
    s = model.model_sampling                                                                  
    start = s.timestep(s.sigma_max)                                                           
    end = s.timestep(s.sigma_min)                                                             
                                                                                              
    if sgm:                                                                                   
        timesteps = torch.linspace(start, end, steps + 1)[:-1]                                
    else:                                                                                     
        timesteps = torch.linspace(start, end, steps)                                         
                                                                                              
    sigs = []                                                                                 
    for x in range(len(timesteps)):                                                           
        ts = timesteps[x]                                                                     
        sigs.append(s.sigma(ts))                                                              
    sigs += [0.0]                                                                             
    return torch.FloatTensor(sigs)

def get_mask_aabb(masks):                                                                                                                                                                                            
    if masks.numel() == 0:                                                                    
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)                      
                                                                                              
    b = masks.shape[0]                                                                        
                                                                                              
    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)                
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)                        
    for i in range(b):                                                                        
        mask = masks[i]                                                                       
        if mask.numel() == 0:                                                                 
            continue                                                                          
        if torch.max(mask != 0) == False:                                                     
            is_empty[i] = True                                                                
            continue                                                                          
        y, x = torch.where(mask)                                                              
        bounding_boxes[i, 0] = torch.min(x)                                                   
        bounding_boxes[i, 1] = torch.min(y)                                                   
        bounding_boxes[i, 2] = torch.max(x)                                                   
        bounding_boxes[i, 3] = torch.max(y)                                                   
                                                                                              
    return bounding_boxes, is_empty

# Copied from ComfyUI's samplers.py
def resolve_areas_and_cond_masks(conditions, h, w, device):                                                                                                                                                          
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):                                                          
        c = conditions[i]                                                                     
        if 'area' in c:                                                                       
            area = c['area']                                                                  
            if area[0] == "percentage":                                                       
                modified = c.copy()                                                           
                area = (max(1, round(area[1] * h)), max(1, round(area[2] * w)), round(area[3] * h), round(area[4] * w))
                modified['area'] = area                                                       
                c = modified                                                                  
                conditions[i] = c                                                             
                                                                                              
        if 'mask' in c:                                                                       
            mask = c['mask']                                                                  
            mask = mask.to(device=device)                                                     
            modified = c.copy()                                                               
            if len(mask.shape) == 2:                                                          
                mask = mask.unsqueeze(0)                                                      
            if mask.shape[1] != h or mask.shape[2] != w:                                      
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
                                                                                              
            if modified.get("set_area_to_bounds", False):                                         
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)                 
                boxes, is_empty = get_mask_aabb(bounds)                                       
                if is_empty[0]:                                                               
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified['area'] = (8, 8, 0, 0)                                           
                else:                                                                         
                    box = boxes[0]                                                            
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])   
                    H = max(8, H)                                                             
                    W = max(8, W)                                                             
                    area = (int(H), int(W), int(Y), int(X))                                   
                    modified['area'] = area                                                   
                                                                                              
            modified['mask'] = mask                                                           
            conditions[i] = modified

# Copied from ComfyUI's samplers.py
def ksampler(sampler_name, extra_options={}, inpaint_options={}):                             
    if sampler_name == "dpm_fast":                                                            
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):           
            sigma_min = sigmas[-1]                                                            
            if sigma_min == 0:                                                                
                sigma_min = sigmas[-2]                                                        
            total_steps = len(sigmas) - 1                                                     
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function                                                  
    elif sampler_name == "dpm_adaptive":                                                      
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable):       
            sigma_min = sigmas[-1]                                                            
            if sigma_min == 0:                                                                
                sigma_min = sigmas[-2]                                                        
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_adaptive_function                                              
    else:                                                                                     
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))    
                                                                                              
    return KSAMPLER(sampler_function, extra_options, inpaint_options)

# Copied from ComfyUI's samplers.py
def wrap_model(model):
    model_denoise = CFGNoisePredictor(model) 
    return model_denoise

# Copied from ComfyUI's samplers.py
def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):                                    
    positive = positive[:]
    negative = negative[:]
    
    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = wrap_model(model)
    
    calculate_start_end_timesteps(model, negative)
    calculate_start_end_timesteps(model, positive)
    
    if latent_image is not None:
        latent_image = model.process_latent_in(latent_image)
    
    if hasattr(model, 'extra_conds'):
        positive = encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        negative = encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        
    #make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)
        
    pre_run_control(model, negative + positive)
    
    apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])
    
    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}
    
    samples = sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    return model.process_latent_out(samples.to(torch.float32))

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]                             
                                                                                              
def calculate_sigmas_scheduler(model, scheduler_name, steps):                                                                                                                                                        
    if scheduler_name == "karras":                                                            
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
    elif scheduler_name == "exponential":                                                     
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
    elif scheduler_name == "normal":                                                          
        sigmas = normal_scheduler(model, steps)                                               
    elif scheduler_name == "simple":                                                          
        sigmas = simple_scheduler(model, steps)                                               
    elif scheduler_name == "ddim_uniform":                                                    
        sigmas = ddim_scheduler(model, steps)                                                 
    elif scheduler_name == "sgm_uniform":                                                     
        sigmas = normal_scheduler(model, steps, sgm=True)                                     
    else:                                                                                     
        print("error invalid scheduler", scheduler_name)                                      
    return sigmas

def sampler_object(name):                                                                     
    if name == "uni_pc":                                                                      
        sampler = UNIPC()                                                                     
    elif name == "uni_pc_bh2":                                                                
        sampler = UNIPCBH2()                                                                  
    elif name == "ddim":                                                                      
        sampler = ksampler("euler", inpaint_options={"random": True})                         
    else:                                                                                     
        sampler = ksampler(name)                                                              
    return sampler                                                                            

class KSamplerBasic:
    """
    Custom sampler based off KSampler class in ComfyUI's samplers.py.
    """
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options
        
        self.model_denoise = CFGNoisePredictor(model)

        # If using a V (velocity) prediction model (SD2.x, etc.), then use a V-denoiser:
        self.model_wrap = comfy.samplers.wrap_model(model)

        self.model_k = KSamplerX0Inpaint(self.model_wrap)
        #print(vars(model))
        #self.sigma_min=float(model.model_sampling.sigma_min)
        #self.sigma_max=float(model.model_sampling.sigma_max)

    # Copied from samplers.py:
    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas_scheduler(self.model, self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas,
                      self.model_options, latent_image=latent_image, denoise_mask=denoise_mask,
                      callback=callback, disable_pbar=disable_pbar, seed=seed)

#        if self.model.is_adm():
#            positive = encode_adm(self.model, positive, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "positive")
#            negative = encode_adm(self.model, negative, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "negative")
#
#        if latent_image is not None:
#            latent_image = self.model.process_latent_in(latent_image)
#
#        extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": self.model_options, "seed":seed}
#
#        cond_concat = None
#        if hasattr(self.model, 'concat_keys'): #inpaint
#            cond_concat = []
#            for ck in self.model.concat_keys:
#                if denoise_mask is not None:
#                    if ck == "mask":
#                        cond_concat.append(denoise_mask[:,:1])
#                    elif ck == "masked_image":
#                        cond_concat.append(latent_image) #NOTE: the latent_image should be masked by the mask in pixel space
#                else:
#                    if ck == "mask":
#                        cond_concat.append(torch.ones_like(noise)[:,:1])
#                    elif ck == "masked_image":
#                        cond_concat.append(blank_inpaint_image_like(noise))
#            extra_args["cond_concat"] = cond_concat
#
#        if sigmas[0] != self.sigmas[0] or (self.denoise is not None and self.denoise < 1.0):
#            max_denoise = False
#        else:
#            max_denoise = True
#
#
#        if self.sampler == "uni_pc":
#            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask, callback=callback, disable=disable_pbar)
#        elif self.sampler == "uni_pc_bh2":
#            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask, callback=callback, variant='bh2', disable=disable_pbar)
#        elif self.sampler == "ddim":
#            timesteps = []
#            for s in range(sigmas.shape[0]):
#                timesteps.insert(0, self.model_wrap.sigma_to_discrete_timestep(sigmas[s]))
#            noise_mask = None
#            if denoise_mask is not None:
#                noise_mask = 1.0 - denoise_mask
#
#            ddim_callback = None
#            if callback is not None:
#                total_steps = len(timesteps) - 1
#                ddim_callback = lambda pred_x0, i: callback(i, pred_x0, None, total_steps)
#
#            sampler = DDIMSampler(self.model, device=self.device)
#            sampler.make_schedule_timesteps(ddim_timesteps=timesteps, verbose=False)
#            z_enc = sampler.stochastic_encode(latent_image, torch.tensor([len(timesteps) - 1] * noise.shape[0]).to(self.device), noise=noise, max_denoise=max_denoise)
#            samples, _ = sampler.sample_custom(ddim_timesteps=timesteps,
#                                                    conditioning=positive,
#                                                    batch_size=noise.shape[0],
#                                                    shape=noise.shape[1:],
#                                                    verbose=False,
#                                                    unconditional_guidance_scale=cfg,
#                                                    unconditional_conditioning=negative,
#                                                    eta=0.0,
#                                                    x_T=z_enc,
#                                                    x0=latent_image,
#                                                    img_callback=ddim_callback,
#                                                    denoise_function=self.model_wrap.predict_eps_discrete_timestep,
#                                                    extra_args=extra_args,
#                                                    mask=noise_mask,
#                                                    to_zero=sigmas[-1]==0,
#                                                    end_step=sigmas.shape[0] - 1,
#                                                    disable_pbar=disable_pbar)
#
#        else:
#            extra_args["denoise_mask"] = denoise_mask
#            self.model_k.latent_image = latent_image
#            self.model_k.noise = noise
#
#            if max_denoise:
#                noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
#            else:
#                noise = noise * sigmas[0]
#
#            k_callback = None
#            total_steps = len(sigmas) - 1
#            if callback is not None:
#                k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)
#
#            if latent_image is not None:
#                noise += latent_image
#            if self.sampler == "dpm_fast":
#                samples = k_diffusion_sampling.sample_dpm_fast(self.model_k, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=k_callback, disable=disable_pbar)
#            elif self.sampler == "dpm_adaptive":
#                samples = k_diffusion_sampling.sample_dpm_adaptive(self.model_k, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=k_callback, disable=disable_pbar)
#            else:
#                samples = getattr(k_diffusion_sampling, "sample_{}".format(self.sampler))(self.model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar)
#
#        return self.model.process_latent_out(samples.to(torch.float32))
