from pipeline import ScanUpdatePipeline

if __name__ == "__main__":
    scinario_type = "03_straight_crawl"
    data_dir = f"dataset/{scinario_type}/pcd/"
    save_dir = f"result/{scinario_type}/"

    pipeline = ScanUpdatePipeline()
    pipeline.get_dataset(data_dir)
    pipeline.run(
        period=10,
        show_trajectory=True,
        show_video=False,
        save_trajectory=False,
        save_video=False,
        save_dir=save_dir,
        analyze=True,
        )