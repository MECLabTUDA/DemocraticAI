from network.ultrasound import get_ultrasound_model
from network.xray import get_xray_model
from network.crc import get_crc_model


def get_model(data_type, model_type, device="cpu"):
    if data_type == "fetalAbdominal":
        return get_ultrasound_model(model_type, device)
    if data_type in ["XRay", "XRayMimic", "XRayMimic200"]:
        return get_xray_model(model_type, device)
    if data_type == "crc":
        return get_crc_model(model_type, device) 

    raise ValueError(f'Unknown data type: {data_type}')