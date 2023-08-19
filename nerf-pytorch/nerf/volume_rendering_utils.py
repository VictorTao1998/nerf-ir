import torch
import torch.nn.functional as F

from .nerf_helpers import cumprod_exclusive, get_minibatches
from .brdf import *

#brdf_specular = specular_pipeline_render_multilight_new
brdf_specular = specular_pipeline_render_multilight_new
import os


def run_network_ir_env(network_fn, pts, surf2c, surf2l, chunksize, embed_fn, embeddirs_fn):
    
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    #c_pts_flat = c_pts.reshape((-1, c_pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    #print(embedded.shape)
    #assert 1==0
    #c_embedded = embed_fn(c_pts_flat)
    #embedded = c_embedded
    #embedded = torch.cat((embedded, c_embedded), dim=-1)
    #print(pts_flat.shape, embedded.shape)

    if embeddirs_fn is not None:
        viewdirs = surf2c[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))

        camdirs = surf2l[..., None, -3:]
        output_dirs = camdirs.expand(pts.shape)
        output_dirs_flat = output_dirs.reshape((-1, output_dirs.shape[-1]))
        #output_dirs_flat = surf2l.reshape((-1, surf2l.shape[-1]))

        embedded_indirs = embeddirs_fn(input_dirs_flat)
        embedded_outdirs = embeddirs_fn(output_dirs_flat)

        embedded = torch.cat((embedded, embedded_indirs, embedded_outdirs), dim=-1)



    #print("before", pts_flat[0,:], c_pts_flat[0,:], viewdirs[0,:], c_viewdirs[0,:])
    #print(pts_flat[0,:],c_pts_flat[0,:],viewdirs[0,:], c_pts_flat.shape)
    #print(embedded.shape)
    #assert 1==0
    batches = get_minibatches(embedded, chunksize=chunksize)
    #print(batches[0].shape)
    #assert 1==0
    preds = [network_fn(batch) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3
):
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :color_channel])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)
    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

def volume_render_radiance_field_ir(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3
):
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :color_channel])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)
    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

def volume_render_radiance_field_ir_env(
    radiance_field,
    #radiance_field_env,
    #radiance_field_env_jitter,
    depth_values,
    ray_origins,
    ray_directions,
    c_ray_directions,
    model_env=None,
    pts=None,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3,
    idx=None,
    #d_n=None,
    joint=False,
    is_env=False,
    #albedo_edit=None,
    #roughness_edit=None,
    #normal_edit=None,
    mode="train",
    logdir=None,
    light_extrinsic=None,
    radiance_backup=None,
    #gt_normal=None,
    encode_position_fn=None,
    encode_direction_fn=None,
):
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    #print(depth_values[0,:], dists[0,:])
    #assert 1==0
    #print(ray_directions[..., None, :].norm(p=2, dim=-1))
    #assert 1==0
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = radiance_field[..., :color_channel]
    #print(rgb[0,0,0], rgb[0,0,1], rgb[0,0,2])
    occupancy = radiance_field[..., color_channel]
    if not torch.all(~torch.isnan(rgb)):
        print("nan rgb!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not torch.all(~torch.isnan(occupancy)):
        print("nan occupancy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    rgb_map = None
    normal_map = None
    albedo_map = None
    roughness_map = None
    normals_diff_map = None
    d_n_map = None
    albedo_smoothness_cost_map = None
    roughness_smoothness_cost_map = None
    normal_smoothness_cost_map = None
    env_rgb = torch.sigmoid(rgb)
    radiance_field_env = None
    surf_brdf = None

    #print(combined_rgb.shape, env_rgb.shape,radiance_field.shape,radiance_field_env.shape)
    #assert 1==0
    noise = 0.0
  

    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        noise = noise.to(radiance_field)

    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise) 
    
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    #print(dists.shape, torch.max(ray_directions[..., None, :].norm(p=2, dim=-1)))
    #assert 1==0
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)# bs x p

    


    env_rgb_map = weights[..., None] * env_rgb
    
    env_rgb_map = env_rgb_map.sum(dim=-2)
    #print(env_rgb_map.shape)
    #assert 1==0
    

    depth_map = weights * depth_values
    #depth_map = depth_weight_f2 * depth_values

    #######################################################################
    depth_map_backup = None
    if radiance_backup is not None:
        with torch.no_grad():
            sigma_a_b = torch.nn.functional.relu(radiance_backup[..., color_channel]) 
            alpha_b = 1.0 - torch.exp(-sigma_a_b * dists)
            weights_b = alpha_b * cumprod_exclusive(1.0 - alpha_b + 1e-10)# bs x p
            depth_map_backup = weights_b * depth_values
            depth_map_backup = depth_map_backup.detach()
            depth_map_backup = depth_map_backup.sum(dim=-1)
    
        #print(depth_map_backup.shape, depth_map.shape)
        #assert 1==0
    #######################################################################

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(torch.max(depth_map))
    
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    max_idx = torch.max(weights,dim=-1).indices # bs x 1
    depth_map_max = depth_values[list(range(depth_values.shape[0])),max_idx]


    if white_background:
        
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)

    if model_env is not None:
        #weights_env = F.softmax(weights, dim=-1) # bs x p
    
        #print(weights.shape, depth_values.shape)
        #assert 1==0
        nlights = 1
   

        if mode == "test":
            if (os.path.exists(os.path.join(logdir, "weights.pt"))):
                weight_save = torch.load(os.path.join(logdir, "weights.pt"))
                depth_values_save = torch.load(os.path.join(logdir, "depth_values.pt"))
                radiance_field_save = torch.load(os.path.join(logdir, "occu.pt"))
                dists_save = torch.load(os.path.join(logdir, "dists.pt"))
                depth_map_save = torch.load(os.path.join(logdir, "depth_map.pt"))
                depth_map_max_save = torch.load(os.path.join(logdir, "depth_map_max.pt"))

                weight_save = torch.cat((weight_save, weights.cpu()), 0)
                depth_values_save = torch.cat((depth_values_save, depth_values.cpu()), 0)
                radiance_field_save = torch.cat((radiance_field_save, radiance_field.cpu()), 0)
                dists_save = torch.cat((dists_save, dists.cpu()), 0)
                depth_map_save = torch.cat((depth_map_save, depth_map.cpu()), 0)
                depth_map_max_save = torch.cat((depth_map_max_save, depth_map_max.cpu()), 0)

                torch.save(weight_save, os.path.join(logdir, "weights.pt"))
                torch.save(depth_values_save, os.path.join(logdir, "depth_values.pt"))
                torch.save(radiance_field_save, os.path.join(logdir, "occu.pt"))
                torch.save(dists_save, os.path.join(logdir, "dists.pt"))
                torch.save(depth_map_save, os.path.join(logdir, "depth_map.pt"))
                torch.save(depth_map_max_save, os.path.join(logdir, "depth_map_max.pt"))

                #print(weight_save.shape, depth_values_save.shape, radiance_field_save.shape, dists_save.shape, depth_map_save.shape, depth_map_max_save.shape)
                #assert 1==0
            else:
                torch.save(weights.cpu(), os.path.join(logdir, "weights.pt"))
                torch.save(depth_values.cpu(), os.path.join(logdir, "depth_values.pt"))
                torch.save(radiance_field.cpu(), os.path.join(logdir, "occu.pt"))
                torch.save(dists.cpu(), os.path.join(logdir, "dists.pt"))
                
                torch.save(depth_map.cpu(), os.path.join(logdir, "depth_map.pt"))
                torch.save(depth_map_max.cpu(), os.path.join(logdir, "depth_map_max.pt"))

            #assert 1==0

  
        rays_o = ray_origins
        rays_d = F.normalize(ray_directions,p=2.0,dim=1)
        surface_z = depth_map
        surface_xyz = rays_o + (surface_z).unsqueeze(-1) * rays_d  # [bs, 3]

        surface_xyz_in = surface_xyz.unsqueeze(-2) # bs x s x 3
        

        
        if joint == True:
            #direct_light, surf2l = model_env.get_light(surface_xyz_in, light_extrinsic) # bs x 3
            direct_light, surf2l = model_env.get_light(pts.detach(), light_extrinsic, surface_xyz.detach()) # bs x 3
            direct_light = torch.sum(direct_light*weights, dim=-1)[...,None]
        else:
            #direct_light, surf2l = model_env.get_light(surface_xyz_in.detach(), light_extrinsic) # bs x 3
            direct_light, surf2l = model_env.get_light(pts.detach(), light_extrinsic, surface_xyz.detach()) # bs x 3
            direct_light = torch.sum(direct_light*(weights.detach()), dim=-1)[...,None]

        

        #surf2l = light_extrinsic[:3,3][None, None, ...] - pts

        
        
        #direct_light = direct_light  # 1024 x 128 x 1 x 1
        

        ray_d_in = rays_d # bs x 3
        surf2c = -ray_d_in
        
        
        radiance_field_env = torch.ones(radiance_field.shape[:2]).unsqueeze(-1).cuda()

        if is_env:
            radiance_field_env = run_network_ir_env(
                model_env,
                pts,  # bs x s x 3
                surf2c, # bs x 3
                surf2l, # bs x 3
                131072,
                encode_position_fn,
                encode_direction_fn,
                
            )

                
        #radiance_field_env = radiance_field_env.squeeze(-1)
        if joint == True:
            surf_brdf = weights[...,None] * radiance_field_env
        else:
            surf_brdf = weights[...,None].detach() * radiance_field_env
        surf_brdf = surf_brdf.sum(dim=-2)

        light_pix_contrib = direct_light

   
        rgb_ir = light_pix_contrib*surf_brdf  # [bs, 1]


        if joint == True:
            rgb_map = env_rgb_map + rgb_ir
        else:
            rgb_map = env_rgb_map.detach() + rgb_ir
        rgb_map = torch.clip(rgb_map,0.,1.)


    out = [rgb_map, env_rgb_map, surf_brdf, disp_map, acc_map, weights, depth_map, depth_map_max, depth_map_backup, sigma_a] + depth_map_dex
    return tuple(out)

def volume_render_reflectance_field(
    reflectance_field,
    SGrender,
    sg_illumination,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3
):
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
    n_sample = reflectance_field.shape[1]
    n_ray = reflectance_field.shape[0]
    basecolor = reflectance_field[...,:3].reshape([-1,3])
    metallic = reflectance_field[...,3:6].reshape([-1,3])
    roughness = reflectance_field[...,6:9].reshape([-1,3])
    normal = reflectance_field[...,9:12].reshape([-1,3])
    alpha = reflectance_field[...,12:15].reshape([-1,3])
    view_dir = ray_directions[...,None,:].expand(-1,n_sample,-1).reshape([-1,3])

    #print(view_dir.shape, ray_directions.shape)

    #print(ray_directions[1,:])
    #assert 1==0

    #print(basecolor.shape, metallic.shape, roughness.shape, normal.shape, alpha.shape, view_dir.shape, sg_illumination.shape)
    output = SGrender(
                sg_illuminations=sg_illumination[None,...],
                basecolor=basecolor,
                metallic=metallic,
                roughness=roughness,
                normal=normal,
                alpha=alpha,
                view_dir=view_dir,
    )
    output = torch.mean(output, dim=-1).reshape([n_ray,n_sample,1])
    #print(output.shape)


    #assert 1==0

    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(output)
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                reflectance_field[..., 15].shape,
                dtype=reflectance_field.dtype,
                device=reflectance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(reflectance_field[..., 15] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)
    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

