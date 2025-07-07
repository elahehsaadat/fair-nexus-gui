from image_utils import load_image_as_array
from nexusformat.nexus import *
import json
import os

def test_create_nexus_file():
    image_array = load_image_as_array("sample_data/microscope1.tif")

    with open("sample_data/metadata.json") as f:
        metadata = json.load(f)

    nxentry = NXentry()
    nxentry.instrument = NXinstrument(name=metadata["instrument"])
    nxentry.sample = NXsample(name=metadata["sample_id"])
    nxentry.data = NXdata(NXfield(image_array, name="image"))
    nxfile = NXroot(nxentry)

    output_path = "tests/test_output.nxs"
    nxfile.save(output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)
