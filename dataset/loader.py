from dataset.FetalAbdominal import FetalAbdominal_loading
from dataset.XRay import XRay_loading
from dataset.XRayMimic import XRayMimic_loading
from dataset.CRC import CRC_loading

def get_data_loader(data_type: str, client_name, num_clients, num_test: float, data_split_seed: int, batch_size=1, shuffle=True):
    if data_type == "fetalAbdominal":
        return FetalAbdominal_loading(client_name, num_clients, num_test, data_split_seed=data_split_seed, batch_size=batch_size, shuffle=shuffle)
    if data_type == "XRay":
        return XRay_loading(client_name, num_clients, num_test, data_split_seed=data_split_seed, batch_size=batch_size, shuffle=shuffle)
    if data_type == "XRayMimic":
        return XRayMimic_loading(client_name, num_clients, num_test, data_split_seed=data_split_seed, batch_size=batch_size, shuffle=shuffle)
    if data_type == "XRayMimic200":
        return XRayMimic_loading(client_name, num_clients, num_test, data_split_seed=data_split_seed, batch_size=batch_size, shuffle=shuffle, version=200)
    if data_type == "crc":
        return CRC_loading(client_name, num_clients, num_test, data_split_seed=data_split_seed, batch_size=batch_size, shuffle=shuffle)

    raise ValueError(f'Unknown data type: {data_type}')
