# %%
import os
import json
from glob import glob
import fire

from app.utils import load_models, batch_audio_detection

# %%
def main(data_root,
         prompt,
         det_out):

    wav_list = glob(os.path.join(data_root, "**/*.wav"), recursive=True)

    det_file = prompt.split("/")[-1].replace("prompt_", "det_").replace(".json", ".csv")

    with open(prompt, 'r') as f:
        prompt_info = json.load(f)

    load_models("CLAP_Jan23")

    batch_audio_detection(wav_list=wav_list, neg_prompts=prompt_info["neg_prompts"],
                          pos_prompts=prompt_info["pos_prompts"],
                          theta=prompt_info["theta"], output_spec=False, output_det=True, 
                          save_path=det_out, det_file=det_file,
                          progress="tqdm")

    print("Detection file saved to {}".format(det_file))

if __name__ == '__main__':
    fire.Fire(main)