from pipeline import ScanUpdatePipeline

if __name__ == "__main__":
    scinario_type = "01_straight_walk"
    data_dir = f"dataset/{scinario_type}/pcd/"
    save_dir = f"result/{scinario_type}/"

    pipeline = ScanUpdatePipeline()
    pipeline.get_dataset(data_dir)
    pipeline.run(
        period=5,
        show_trajectory=True,
        show_video=True,
        save_trajectory=True,
        save_video=True,
        save_dir=save_dir,
        )