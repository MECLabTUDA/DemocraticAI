import os
import getpass
def get_root_path():
    root_dir = "/local/scratch/"
    if not os.path.exists(root_dir):
        root_dir = "/gris/scratch-gris-filesrv/"
    return root_dir



# data paths
xray_mimic_path = os.path.join(get_root_path(), "clmn1/data/xray/MIMIC_50")
xray_mimic200_path = os.path.join(get_root_path(), "clmn1/data/xray/MIMIC_200")
xray_path = os.path.join(get_root_path(), "clmn1/data/xray")
ultrasound_path = os.path.join(get_root_path(), "clmn1/data/ultrasound/4gcpm9dsc3-1/FetalAbdominalStructuresSegmentationDatasetUsingUltrasonicImages")
crc_path = os.path.join(get_root_path(), "CRC")


# training results paths
if getpass.getuser() == "mikonsta":
    results_workingdir_path = os.path.join(get_root_path(), "mikonsta/fednca_journal/workingdir4")
    results_single_path = os.path.join(get_root_path(), "mikonsta/fednca_journal/single")
    results_reconstructions_path = os.path.join(get_root_path(), "mikonsta/fednca_journal/reconstructions")
elif getpass.getuser() == "nlemke":
    results_workingdir_path = os.path.join(get_root_path(), "clmn1/fednca_journal/workingdir")
    results_single_path = os.path.join(get_root_path(), "clmn1/fednca_journal/single")
    results_reconstructions_path = os.path.join(get_root_path(), "clmn1/fednca_journal/reconstructions")
else:
    raise ValueError("User not recognized, please add user-specific paths to utils/root_path.py")


results_tempdir_path = os.path.join(results_workingdir_path, "tempdir") # temporary directory for storing additional files like random initialization, private keys, etc.
os.makedirs(results_reconstructions_path, exist_ok=True)