import re


def find_template_parameters(kernel_src: str) -> bool:
    """Checks if the kernel is templated, and if yes, stores the arguments in the kernel_build_dir"""
    # TODO: currently via simple regular expression, would be better via clang
    if "forward_templated" in kernel_src:
        matches = re.findall(r"forward_templated<([^>]*)>", kernel_src)

        # extract all supported combinations and cast to correct datatypes
        supported_combinations = []
        for m in matches:
            params_as_str = m.split(",")
            param_tuple = []
            for param in params_as_str:
                try:
                    param_as_float = float(param)
                    if "." not in param:
                        # cast to int
                        param_tuple.append(int(param_as_float))
                    else:
                        # convert to float
                        param_tuple.append(param_as_float)
                except ValueError:
                    # use string
                    stripped = param.strip()
                    if stripped in ["true", "false"]:
                        param_tuple.append(stripped == "true")
                    else:
                        param_tuple.append(stripped)
            supported_combinations.append(tuple(param_tuple))

        # # simple version only supporting int:
        # supported_combinations = [tuple(map(int, m.split(","))) for m in matches]
        print("Templated kernel! Supported combinations:", supported_combinations)
        return supported_combinations
    return None
