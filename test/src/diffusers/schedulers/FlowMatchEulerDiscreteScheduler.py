import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
from utils import TestCase


class TestDiffusersFlowMatchEulerDiscreteScheduler(TestCase):
    def test_schedule(self):
        # Default scheduler config: static shift of 1.0, so mu is unused.
        num_inference_steps = 4
        image_seq_len = 6

        mu = compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        scheduler.set_timesteps(num_inference_steps, mu=mu)

        actual = self.cli(
            "FlowMatchEulerDiscreteScheduler_schedule",
            "--num_inference_steps", str(num_inference_steps),
            "--mu", str(mu),
        )

        self.assertTensors(actual, [scheduler.timesteps, scheduler.sigmas])

    def test_schedule_dynamic_shifting(self):
        # Resolution-dependent shifting: base sigmas are shifted by mu,
        # which is computed from the generated image sequence length.
        num_inference_steps = 4
        image_seq_len = 4096

        mu = compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)

        scheduler.set_timesteps(num_inference_steps, mu=mu)

        actual = self.cli(
            "FlowMatchEulerDiscreteScheduler_schedule",
            "--num_inference_steps", str(num_inference_steps),
            "--mu", str(mu),
            "--use_dynamic_shifting", "true",
        )

        self.assertTensors(actual, [scheduler.timesteps, scheduler.sigmas])

    def test_schedule_single_step(self):
        num_inference_steps = 1
        image_seq_len = 6

        mu = compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        scheduler.set_timesteps(num_inference_steps, mu=mu)

        actual = self.cli(
            "FlowMatchEulerDiscreteScheduler_schedule",
            "--num_inference_steps", str(num_inference_steps),
            "--mu", str(mu),
        )

        self.assertTensors(actual, [scheduler.timesteps, scheduler.sigmas])

    def test_step(self):
        # Euler transition: C++ derives dt from its own schedule,
        # mirroring the pipeline denoising loop.
        num_inference_steps = 5
        image_seq_len = 6
        index = 2

        mu = compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)

        scheduler.set_timesteps(num_inference_steps, mu=mu)

        generator = torch.Generator()

        model_output = torch.randn(1, 6, 16, generator=generator)
        sample = torch.randn(1, 6, 16, generator=generator)

        expected = scheduler.step(
            model_output,
            scheduler.timesteps[index],
            sample,
            return_dict=False,
        )[0]

        actual = self.cli(
            "FlowMatchEulerDiscreteScheduler_step",
            "--num_inference_steps", str(num_inference_steps),
            "--mu", str(mu),
            "--index", str(index),
            "--model_output", str(model_output.tolist()),
            "--sample", str(sample.tolist()),
            "--use_dynamic_shifting", "true",
        )

        self.assertTensors(actual, [expected])
