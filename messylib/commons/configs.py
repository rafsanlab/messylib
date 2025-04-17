import os, json


class Configs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_param(self, key, value):
        setattr(self, key, value)

    def save_to_json(self, filepath):

        # function to check if value is tuple then return a dict
        def convert(obj):
            if isinstance(obj, tuple):
                return {"__tuple__": True, "items": list(obj)}
            # elif isinstance(obj, nn.Module):
            #     obj = str(obj)
            # elif isinstance(obj, torch.device):
            #     obj = str(obj)
            return obj

        # add nested dict for tuple values
        data = {key: convert(value) for key, value in self.__dict__.items()}

        if filepath.endswith(".json"):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        else:
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(filepath, "config.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
            print(f"Config file save at : {filepath}")

    def load_from_json(self, file_path):

        # function to check if values is tuples
        def custom_decoder(obj):
            if "__tuple__" in obj:
                return tuple(obj["items"])
            return obj

        # open data with tuple converter
        with open(file_path, "r") as file:
            params = json.load(file, object_hook=custom_decoder)

        for key, value in params.items():
            setattr(self, key, value)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)
