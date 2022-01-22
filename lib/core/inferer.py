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
            (tbd_ls, tbd_le, tbd_gs, tbd_ge), imr_out = model(video_data)

        for jdx in range(batch_size):
            # get snippet info
            video_name = video_info["video_name"][jdx]
            video_snippets = video_info["indices"][jdx].numpy()
            start = video_snippets[0]
            end = video_snippets[-1]

            # detach result
            pred_local_start = tbd_ls[jdx].cpu().detach().numpy()
            pred_local_end = tbd_le[jdx].cpu().detach().numpy()
            pred_global_start = tbd_gs[jdx].cpu().detach().numpy()
            pred_global_end = tbd_ge[jdx].cpu().detach().numpy()
            pred_iou_map = imr_out[jdx].cpu().detach().numpy()

            result = [
                video_snippets,
                pred_local_start,
                pred_local_end,
                pred_global_start,
                pred_global_end,
                pred_iou_map,
            ]   

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
