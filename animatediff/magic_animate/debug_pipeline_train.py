    def train(
            self,
            prompt: Union[str, List[str]],
            prompt_embeddings: Optional[torch.FloatTensor] = None,
            video_length: Optional[int] = 8,
            height: Optional[int] = None,
            width: Optional[int] = None,
            timestep: Union[torch.Tensor, float, int] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator,
                                      List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            callback_steps: Optional[int] = 1,
            controlnet_condition: list = None,
            controlnet_conditioning_scale: float = 1.0,
            context_batch_size: int = 1,
            init_latents: Optional[torch.FloatTensor] = None,
            appearance_encoder=None,
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
        if init_latents is not None:
            batch_size = init_latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = False

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
            text_embeddings = torch.cat(
                [prompt_embeddings] * context_batch_size)

        reference_control_writer = ReferenceAttentionControl(appearance_encoder, do_classifier_free_guidance=False,
                                                             mode='write', batch_size=1)
        reference_control_reader = ReferenceAttentionControl(self.unet, do_classifier_free_guidance=False, mode='read',
                                                             batch_size=1)

        # Prepare video
        # FIXME: verify if num_videos_per_prompt > 1 works
        assert num_videos_per_prompt == 1
        # assert batch_size == 1  # FIXME: verify if batch_size > 1 works
        control = self.prepare_condition(
            condition=controlnet_condition,
            device=device,
            dtype=controlnet.dtype,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        controlnet_cond_images = control

        # Prepare latent variables
        if init_latents is None:
            # latents = rearrange(init_latents, "(b f) c h w -> b c f h w", f=video_length)
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
            )
        else:
            latents = init_latents

        latents_dtype = latents.dtype

        # Prepare text embeddings for controlnet
        controlnet_text_embeddings_c = text_embeddings.repeat_interleave(
            video_length, 0)

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

        t = timestep

        """
        ref_image_latents torch.Size([2, 4, 64, 64])                                                                                                                                                  │····················
        text_embeddings torch.Size([1, 77, 768]) 
        """
        appearance_encoder(
            ref_image_latents,
            t,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )

        # prepare inputs for controlnet
        controlnet_latent_input = rearrange(
            latents, "b c f h w -> (b f) c h w")

        # controlnet inference
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            controlnet_latent_input,
            t,
            encoder_hidden_states=controlnet_text_embeddings_c,
            controlnet_cond=controlnet_cond_images,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )

        reference_control_reader.update(reference_control_writer)

        # reshape controlnet conditions
        mid_block_res_sample = rearrange(
            mid_block_res_sample, '(b f) c h w -> b c f h w', f=video_length)
        new_down_block_res_samples = ()
        for down_idx, down_sample in enumerate(down_block_res_samples):
            down_sample = rearrange(
                down_sample, '(b f) c h w -> b c f h w', f=video_length)
            new_down_block_res_samples = new_down_block_res_samples + \
                (down_sample, )
        down_block_res_samples = new_down_block_res_samples

        # predict the noise residual
        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        reference_control_reader.clear()
        reference_control_writer.clear()

        return noise_pred