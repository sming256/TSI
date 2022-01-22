import torch
import os
import tqdm
import pickle


def inference(model, data_loader, logger, cfg):
    output_path = "./exps/{}/output/".format(cfg.EXP_NAME)

    for video_info, video_data in tqdm.tqdm(data_loader):
        video_data = video_data.cuda()
        batch_size = video_data.shape[0]

        with torch.no_grad():
            (tem_s, tem_e), pem_out = model(video_data)

        for jdx in range(batch_size):
            # get snippet info
            video_name = video_info["video_name"][jdx]
            video_snippets = video_info["indices"][jdx].numpy()
            start = video_snippets[0]
            end = video_snippets[-1]

            # detach result
            pred_start = tem_s[jdx].cpu().detach().numpy()
            pred_end = tem_e[jdx].cpu().detach().numpy()
            pred_iou_map = pem_out[jdx].cpu().detach().numpy()

            result = [video_snippets, pred_start, pred_end, pred_iou_map]

            # save result
            if cfg.DATASET.name in ["anet_1_3", "hacs"]:
                file_path = os.path.join(output_path, "{}.pkl".format(video_name))
            elif cfg.DATASET.name == "thumos_14":
                output_folder = os.path.join(output_path, video_name)
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                file_path = os.path.join(output_folder, "{}_{}.pkl".format(start, end))

            with open(file_path, "wb") as outfile:
                pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)
