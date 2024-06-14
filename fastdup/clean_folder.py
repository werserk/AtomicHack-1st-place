import fastdup
fd = fastdup.create(work_dir="/home/clean_data/",
                    input_dir="/home/data/")
fd.run(ccthreshold=0.1)

fd.summary()

fd.vis.duplicates_gallery(num_images=500)
