import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)

class OpenAIWrapper(IdentityWrapper):
    def forward(self, x, t, c, **kwargs):
        """
        Args:
            x: input tensor
            t: timestep
            c: conditioning dictionary
        """
        # Ensure input tensors are float32
        x = x.to(torch.float32)
        t = t.to(torch.float32)
        
        if isinstance(c, dict):
            # Convert all tensors in conditioning dict to float32
            for k, v in c.items():
                if isinstance(v, torch.Tensor):
                    c[k] = v.to(torch.float32)
            
            # Extract and process conditioning
            concat = c.get("concat", torch.Tensor([]).type_as(x))
            context = c.get("crossattn", None)
            y = c.get("vector", None)
            
            # Concatenate if concat tensor exists
            if concat.numel() > 0:
                x = torch.cat((x, concat), dim=1)
            
            # Return with proper conditioning
            if "cond_view" in c:
                return self.diffusion_model(
                    x,
                    timesteps=t,
                    context=context,
                    y=y,
                    cond_view=c.get("cond_view"),
                    cond_motion=c.get("cond_motion"),
                    **kwargs
                )
            else:
                return self.diffusion_model(
                    x,
                    timesteps=t,
                    context=context,
                    y=y,
                    **kwargs
                )
        else:
            # Handle legacy case where c is not a dict
            return self.diffusion_model(x, timesteps=t, **kwargs)
        
# class OpenAIWrapper(IdentityWrapper):
#     def forward(
#         self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
#     ) -> torch.Tensor:
#         x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
#         if "cond_view" in c:
#             return self.diffusion_model(
#                 x,
#                 timesteps=t,
#                 context=c.get("crossattn", None),
#                 y=c.get("vector", None),
#                 cond_view=c.get("cond_view", None),
#                 cond_motion=c.get("cond_motion", None),
#                 **kwargs,
#             )
#         else:
#             return self.diffusion_model(
#                 x,
#                 timesteps=t,
#                 context=c.get("crossattn", None),
#                 y=c.get("vector", None),
#                 **kwargs,
#             )
