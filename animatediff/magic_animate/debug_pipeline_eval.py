@torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int],
            prompt_embeddings: Optional[torch.FloatTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator,
                                      List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[
                int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            controlnet_condition: list = None,
            controlnet_conditioning_scale: float = 1.0,
            context_frames: int = 16,
            context_stride: int = 1,
            context_overlap: int = 4,
            context_batch_size: int = 1,
            context_schedule: str = "uniform",
            init_latents: Optional[torch.FloatTensor] = None,
            num_actual_inference_steps: Optional[int] = None,
            appearance_encoder=None,
            reference_control_writer=None,
            reference_control_reader=None,
            source_image: str = None,
            decoder_consistency=None,
            **kwargs,
    ):
        """
        New args:
        - controlnet_condition          : condition map (e.g., depth, canny, keypoints) for controlnet
        - controlnet_conditioning_scale : conditioning scale for controlnet
        - init_latents                  : initial latents to begin with (used along with invert())
        - num_actual_inference_steps    : number of actual inference steps (while total steps is num_inference_steps)
        """
        controlnet = self.controlnet

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        if prompt_embeddings is None:
            prompt = prompt if isinstance(prompt, list) else [
                prompt] * batch_size
            if negative_prompt is not None:
                negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [
                    negative_prompt] * batch_size
            text_embeddings = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            text_embeddings = torch.cat([text_embeddings] * context_batch_size)
        else:
            # project from (batch_size, 257, 1280) to (batch_size, 16, 768)
            with torch.inference_mode():
                prompt_embeddings = self.unet.image_proj_model(
                    prompt_embeddings)
            text_embeddings = torch.cat(
                [prompt_embeddings] * context_batch_size)

        reference_control_writer = ReferenceAttentionControl(appearance_encoder,
                                                             do_classifier_free_guidance=True,
                                                             mode='write',
                                                             batch_size=context_batch_size,
                                                             clip_length=context_frames)
        reference_control_reader = ReferenceAttentionControl(self.unet,
                                                             do_classifier_free_guidance=True,
                                                             mode='read',
                                                             batch_size=context_batch_size,
                                                             clip_length=context_frames)

        is_dist_initialized = kwargs.get("dist", False)
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)

        # Prepare video
        # FIXME: verify if num_videos_per_prompt > 1 works
        assert num_videos_per_prompt == 1
        assert batch_size == 1  # FIXME: verify if batch_size > 1 works
        control = self.prepare_condition(
            condition=controlnet_condition,
            device=device,
            dtype=controlnet.dtype,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        controlnet_uncond_images, controlnet_cond_images = control.chunk(2)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        if init_latents is not None:
            latents = rearrange(
                init_latents, "(b f) c h w -> b c f h w", f=video_length)
        else:
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
                clip_length=context_frames
            )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare text embeddings for controlnet
        controlnet_text_embeddings = text_embeddings.repeat_interleave(
            video_length, 0)
        _, controlnet_text_embeddings_c = controlnet_text_embeddings.chunk(2)

        controlnet_res_samples_cache_dict = {
            i: None for i in range(video_length)}

        # For img2img setting
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps

        if isinstance(source_image, str):
            ref_image_latents = self.images2latents(np.array(Image.open(source_image).resize((width, height)))[None, :],
                                                    latents_dtype).to(device)
        elif isinstance(source_image, np.ndarray):
            ref_image_latents = self.images2latents(
                source_image[None, :], latents_dtype).to(device)
        else:
            source_image = rearrange(source_image, "f h w c -> f c h w").to(device)
            ref_image_latents = []
            for frame_idx in range(source_image.shape[0]):
                ref_image_latents.append(self.vae.encode(
                    source_image[frame_idx:frame_idx + 1])['latent_dist'].mean * 0.18215)
            ref_image_latents = torch.cat(ref_image_latents)
        # FIX
        if batch_size == 1:
            ref_image_latents = ref_image_latents[:1]
        context_scheduler = get_context_scheduler(context_schedule)

        # Denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=(rank != 0)):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            noise_pred = torch.zeros(
                (latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                 *latents.shape[1:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            counter = torch.zeros(
                (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
            )

            appearance_encoder(
                ref_image_latents.repeat(
                    context_batch_size * (2 if do_classifier_free_guidance else 1), 1, 1, 1),
                t,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )

            context_queue = list(context_scheduler(
                0, num_inference_steps, latents.shape[2], context_frames, context_stride, 0
            ))
            num_context_batches = math.ceil(
                len(context_queue) / context_batch_size)
            for i in range(num_context_batches):
                context = context_queue[i *
                                        context_batch_size: (i + 1) * context_batch_size]
                # expand the latents if we are doing classifier free guidance
                controlnet_latent_input = (
                    torch.cat([latents[:, :, c] for c in context])
                    .to(device)
                )
                controlnet_latent_input = self.scheduler.scale_model_input(
                    controlnet_latent_input, t)

                # prepare inputs for controlnet
                b, c, f, h, w = controlnet_latent_input.shape
                controlnet_latent_input = rearrange(
                    controlnet_latent_input, "b c f h w -> (b f) c h w")

                # controlnet inference
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_input,
                    t,
                    encoder_hidden_states=torch.cat(
                        [controlnet_text_embeddings_c[c] for c in context]),
                    controlnet_cond=torch.cat(
                        [controlnet_cond_images[c] for c in context]),
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                for j, k in enumerate(np.concatenate(np.array(context))):
                    controlnet_res_samples_cache_dict[k] = (
                        [sample[j:j + 1] for sample in down_block_res_samples], mid_block_res_sample[j:j + 1])

            context_queue = list(context_scheduler(
                0, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
            ))

            num_context_batches = math.ceil(
                len(context_queue) / context_batch_size)
            global_context = []
            for i in range(num_context_batches):
                global_context.append(
                    context_queue[i * context_batch_size: (i + 1) * context_batch_size])

            for context in global_context[rank::world_size]:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents[:, :, c] for c in context])
                    .to(device)
                    .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                b, c, f, h, w = latent_model_input.shape
                down_block_res_samples, mid_block_res_sample = self.select_controlnet_res_samples(
                    controlnet_res_samples_cache_dict,
                    context,
                    do_classifier_free_guidance,
                    b, f
                )

                reference_control_reader.update(reference_control_writer)

                # predict the noise residual
                pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings[:b],
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                reference_control_reader.clear()

                pred_uc, pred_c = pred.chunk(2)
                pred = torch.cat([pred_uc.unsqueeze(0), pred_c.unsqueeze(0)])
                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + pred[:, j]
                    counter[:, :, c] = counter[:, :, c] + 1

            if is_dist_initialized:
                noise_pred_gathered = [torch.zeros_like(
                    noise_pred) for _ in range(world_size)]
                if rank == 0:
                    dist.gather(tensor=noise_pred,
                                gather_list=noise_pred_gathered, dst=0)
                else:
                    dist.gather(tensor=noise_pred, gather_list=[], dst=0)
                dist.barrier()

                if rank == 0:
                    for k in range(1, world_size):
                        for context in global_context[k::world_size]:
                            for j, c in enumerate(context):
                                noise_pred[:, :, c] = noise_pred[:, :,
                                                                 c] + noise_pred_gathered[k][:, :, c]
                                counter[:, :, c] = counter[:, :, c] + 1

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = (
                    noise_pred / counter).chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if is_dist_initialized:
                dist.broadcast(latents, 0)
                dist.barrier()

            reference_control_writer.clear()

        interpolation_factor = 1
        latents = self.interpolate_latents(
            latents, interpolation_factor, device)
        # Post-processing
        video = self.decode_latents(
            latents, rank, decoder_consistency=decoder_consistency)

        if is_dist_initialized:
            dist.barrier()

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)