import megfile
from caption_coca            import worker as worker_caption_coca
from caption_videoblip       import worker as worker_caption_videoblip
from motion_score            import worker as worker_motion_score
from aesthetics_score        import worker as worker_aesthetic
from text_image_similarities import worker as worker_similarity
from text_detection          import worker as worker_text_detection
from loguru import logger
import jsonlines

def pipeline(video_s3path_list, out_jsonl_dir, cache_dir="/dev/shm/"):

    logger.info("run cpation (CoCa) ...")
    caption_coca_dic          = worker_caption_coca(
        video_s3path_list, 
        megfile.smart_path_join(out_jsonl_dir, "caption_coca"), 
        megfile.smart_path_join(cache_dir, "caption_coca"))
    logger.info(f"run cpation (CoCa) done.")

    logger.info("run cpation (videoblip) ...")
    caption_videoblip_dic     = worker_caption_videoblip(
        video_s3path_list, 
        megfile.smart_path_join(out_jsonl_dir, "caption_videoblip"), 
        megfile.smart_path_join(cache_dir, "caption_videoblip"))
    logger.info("run cpation (videoblip) done. ")

    logger.info("run motionscore (Farneback) ...")
    caption_motionscore_dic   = worker_motion_score(
        video_s3path_list, 
        megfile.smart_path_join(out_jsonl_dir, "motionscore"), 
        megfile.smart_path_join(cache_dir, "motionscore"))
    logger.info("run motionscore (Farneback) done. ")

    logger.info("run aesthetic score ... ")
    aesthetic_dic    = worker_aesthetic(
        video_s3path_list, 
        megfile.smart_path_join(out_jsonl_dir, "aesthetic"), 
        megfile.smart_path_join(cache_dir, "aesthetic"))
    logger.info("run aesthetic score done. ")

    logger.info("run text-image similarity (CoCa) ... ")
    similarity_coca_dic    = worker_similarity(
        video_s3path_list, 
        caption_coca_dic,
        megfile.smart_path_join(out_jsonl_dir, "text_image_similarity_coca"), 
        megfile.smart_path_join(cache_dir, "text_image_similarity_coca"))
    logger.info("run text-image similarity (CoCa) done. ")

    logger.info("run text-image similarity (VideoBLIP) ... ")
    similarity_videoblip_dic    = worker_similarity(
        video_s3path_list, 
        caption_videoblip_dic,
        megfile.smart_path_join(out_jsonl_dir, "text_image_similarity_videoblip"), 
        megfile.smart_path_join(cache_dir, "text_image_similarity_videoblip"))
    logger.info("run text-image similarity (VideoBLIP) done. ")

    logger.info("run textdetection (Craft) ... ")
    caption_textdetection_dic = worker_text_detection(
        video_s3path_list, 
        megfile.smart_path_join(out_jsonl_dir, "textdetection"), 
        megfile.smart_path_join(cache_dir, "textdetection"))
    logger.info("run textdetection (Craft) done. ")

    ret_dic = {}
    for video_s3path in video_s3path_list:
            ret_dic[video_s3path] = {
                "video_path"                      : video_s3path,
                "caption_coca"                    : caption_coca_dic[video_s3path],
                "caption_videoblip"               : caption_videoblip_dic[video_s3path],
                "motionscore"                     : caption_motionscore_dic[video_s3path],
                "aesthetic"                       : aesthetic_dic[video_s3path],
                "text_image_similarity_coca"      : similarity_coca_dic[video_s3path],
                "text_image_similarity_videoblip" : similarity_videoblip_dic[video_s3path],
                "text_detection"                  : caption_textdetection_dic[video_s3path],
            }

    return ret_dic


if __name__ == "__main__":
    import argparse
    import torch
    import resource

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="s3://a-collections-sh/pexels/video")
    parser.add_argument('--out_dir', type=str, default="s3://data-transfer-tos-shanghai-818/midjourney/jmh/Video/tempfile/tag_pipeline")
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()    

    mp4path_list = list(megfile.smart_glob(megfile.smart_path_join(args.video_dir,"*.mp4")))[args.start_idx: args.end_idx]
    cache_dir = "/dev/shm/tag_pipeline"
    megfile.smart_makedirs(cache_dir, exist_ok=True)
    metainfo_dic = pipeline(mp4path_list, args.out_dir, cache_dir=cache_dir)
    
    local_jsonl_path = megfile.smart_path_join(cache_dir, f"svd_metainfo.{args.start_idx}-{args.end_idx}.jsonl")
    with jsonlines.open(local_jsonl_path, mode="w") as file_jsonl:
        for k, v in metainfo_dic.items():
            file_jsonl.write(v)
    out_jsonl_path = megfile.smart_path_join(args.out_dir, f"svd_metainfo.{args.start_idx}-{args.end_idx}.jsonl")
    megfile.smart_move(local_jsonl_path, out_jsonl_path) 
    print("done .")    

    print(f"RAM PEAK: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3)}")
    print(f"VRAM PEAK: {torch.cuda.max_memory_allocated()/(1024**3)}")