import fastdup
fd = fastdup.create(work_dir="/home/clean_data/",
                    input_dir="/home/data/")
fd.run(ccthreshold=0.1)

fd.summary()

fd.vis.duplicates_gallery(num_images=500)
fd.vis.outliers_gallery(num_images=500)
fd.vis.component_gallery(num_images=500)

connected_components_df , _ = fd.connected_components()
connected_components_df.to_csv('/home/clean_data/connected_components_df.csv')


def get_clusters(df, sort_by='count', min_count=2, ascending=False):
    # columns to aggregate
    agg_dict = {'img_filename': list, 'mean_distance': max, 'count': len}

    if 'label' in df.columns:
        agg_dict['label'] = list

    # filter by count
    df = df[df['count'] >= min_count]

    # group and aggregate columns
    grouped_df = df.groupby('component_id').agg(agg_dict)

    # sort
    grouped_df = grouped_df.sort_values(by=[sort_by], ascending=ascending)
    return grouped_df

clusters_df = get_clusters(connected_components_df)
clusters_df.to_csv('/home/clean_data/clusters_df.csv')
