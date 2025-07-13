"""
The generation pipeline.
"""

from typing import List, Optional, Tuple, Union, Dict
import torch
from .condition import ImagingCondition
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import ImagePipelineOutput, DDPMPipeline


class DDPMCondPipeline(DDPMPipeline):
    @torch.enable_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "numpy",
        condition_input: Dict[str, torch.Tensor] = None,
        conditioner: ImagingCondition = None,
        step_size: float = 400.,
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Conditional generation for PD/T1/T2 estimation.

        :param batch_size: Number of posterior samples.
        :param generator: A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
        :param num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
        :param output_type: (`str`, *optional*, defaults to `"pil"`)
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        :param condition_input: The observed image.
        :param conditioner: An image condition object that evaluates the log-likelihood.
        :param step_size: Controls the strength of log-likelihood, ξ in the paper.
        :param return_dict: Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        :return:
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        image.requires_grad = True

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image,
                                        condition_input=condition_input,
                                        conditioner=conditioner,
                                        step_size=step_size,
                                        generator=generator).prev_sample
        return ImagePipelineOutput(images=image.detach())



