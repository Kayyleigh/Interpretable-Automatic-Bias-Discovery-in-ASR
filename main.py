import json
import seaborn as sns

from python.clustering.data_prep import DataPrep
from python.clustering.grouping import Clustering, PredefinedGrouping
from python.clustering.visualization import DataVisualization, GroupingVisualization
from python.evaluation.filepaths import FilepathManager 
from python.evaluation.asr_performance_data import AsrPerformanceData 
from python.evaluation.grouping_performance import GroupingPerformance
from python.evaluation.metrics import DiffBias, RelDiffBias, OverallBias 
from python.evaluation.bias_visualization import BiasVisualization
from python.comparison.subsets_visualization import SubsetsVisualization
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cluster import AgglomerativeClustering

def bias_metric_pipeline(file_path, speaking_style, custom_features, excluded_features=[]):

    # define different color palettes to avoid confusion
    cluster_colors = sns.color_palette("colorblind", 10)[5:]
    demographic_colors = sns.color_palette("colorblind", 5)

    # Prepare dataframe of extracted features
    data_prep = DataPrep(file_path, speaking_style=speaking_style, 
                         feature_dict=custom_features, dropcolumns=excluded_features)
    data_prep.normalize_data(scaler=MinMaxScaler(), check_normality=False)

    # Initialize clustering algorithm
    clusteringAlgorithm = AgglomerativeClustering(n_clusters=5, linkage='ward')

    # Apply clustering to feature vectors of extracted acoustic and prosodic features
    clustering = Clustering(data_prep, clusteringAlgorithm)
    cluster_sizes = clustering.get_group_sizes()

    print(cluster_sizes)

    # Enable visualization of extracted features and resulting clusters
    data_visualizer = DataVisualization(data_prep)
    cluster_visualizer = GroupingVisualization(clustering, palette=cluster_colors)

    # Plot correlation matrix
    data_visualizer.correlation_matrix()

    # Plot clustering in a 2d view (axes should match labels used in the .csv of extracted features)
    #cluster_visualizer.show_grouping_2d("Mean Pitch (Hz)", "Mean Articulation Rate (phpm)")

    # Show boxplots of clusters
    cluster_visualizer.show_grouping_boxplots()

    # Prepare ASR performance data
    # Right now these assume WER
    filepath_manager = FilepathManager('python\config\config.json')
    asr_performance_data = AsrPerformanceData(filepath_manager=filepath_manager, speaking_style_id=speaking_style)

    # Instantiate class where extracted features and ASR performance are merged
    clustering_performance = GroupingPerformance(clustering, asr_performance_data)

    # Print cluster-to-demographic speaker group assignment distributions (percentages)
    print(clustering_performance.get_group_distribution("Cluster", "Group"))
    print(clustering_performance.get_group_distribution("Group", "Cluster"))

    # Print cluster-to-demographic speaker group assignment distributions (counts)
    print(clustering_performance.get_group_counts_comparison("Cluster", "Group"))
    print(clustering_performance.get_group_counts_comparison("Group", "Cluster"))

    # Group by demographic information    
    numeric_cols = clustering_performance.get_data().select_dtypes(include='float').columns
    agg_dict = {col: 'mean' for col in numeric_cols}
    agg_dict['Cluster'] = 'first'   # Cluster consistent within speaker 
    agg_dict['Group'] = 'first'     # Group consistent within speaker 
    combined_data_by_speaker = clustering_performance.get_data().groupby('Filename').agg(agg_dict)
    demographic_groups = combined_data_by_speaker['Group'].tolist()

    # Create grouping object (similar to clustering) to visualize extracted features for demographic groups
    demographic_grouping = PredefinedGrouping(data_prep=data_prep, group_labels=demographic_groups)
    group_sizes = demographic_grouping.get_group_sizes()
    
    # Print number of speakers per demographic group
    print(group_sizes)

    # Create visualizer for the demographic grouping
    demographic_visualizer = GroupingVisualization(demographic_grouping, palette=demographic_colors)
    
    # Plot 2d view of demogoraphic groups with respect to 2 of the features
    demographic_visualizer.show_grouping_2d('Mean Articulation Rate (phpm)','Mean Pitch (Hz)')

    # Show boxplots of demographic groups
    demographic_visualizer.show_grouping_boxplots()

    # Instantiate a Difference Bias measure
    diff_bias_measure = DiffBias()

    # Instantiate a Relative Difference Bias measure
    rel_diff_bias_measure = RelDiffBias()

    # Instantiate corresponding Overall Bias measures
    overall_bias_measure = OverallBias(diff_bias_measure)
    rel_overall_bias_measure = OverallBias(rel_diff_bias_measure)

    # Calculate Overall Difference Bias for the ASR models for demographic groups
    overall_bias_groups = clustering_performance.calculate_bias('Group', overall_bias_measure)

    # Calculate Overall Difference Bias for the ASR models for clusters
    overall_bias_clusters = clustering_performance.calculate_bias('Cluster', overall_bias_measure)

    # Calculate Overall Relative Difference Bias for the ASR models for demographic groups
    rel_overall_bias_groups = clustering_performance.calculate_bias('Group', rel_overall_bias_measure)

    # Calculate Overall Relative Difference Bias for the ASR models for clusters
    rel_overall_bias_clusters = clustering_performance.calculate_bias('Cluster', rel_overall_bias_measure)

    # Print Overall Difference Biases 
    print(f"Overall difference bias (Groups) for {speaking_style}: {overall_bias_groups}")
    print(f"Overall difference bias (Clusters) for {speaking_style}: {overall_bias_clusters}")

    # Print Overall Relative Difference Biases 
    print(f"Overall relative difference bias (Groups) for {speaking_style}: {rel_overall_bias_groups}")
    print(f"Overall relative difference bias (Clusters) for {speaking_style}: {rel_overall_bias_clusters}")

    # Get Word Error Rate per ASR model per demographic group
    error_rates_per_group = clustering_performance.get_group_WERs('Group')
    # Get Word Error Rate per ASR model per cluster
    error_rates_per_cluster = clustering_performance.get_group_WERs('Cluster')

    # Print WERs per model per group/cluster
    print(error_rates_per_group.pivot_table(values='WER', index='Group', columns='Model'))
    print(error_rates_per_cluster.pivot_table(values='WER', index='Cluster', columns='Model'))

    # Instantiate Bias visualizer
    bias_visualizer = BiasVisualization(data_processor=clustering_performance)

    # Plot Overall Difference Bias per model per demographic group 
    bias_visualizer.plot_group_biases('Group', overall_bias_measure)

    # Plot Overall Difference Bias per model per cluster 
    bias_visualizer.plot_group_biases('Cluster', overall_bias_measure)
    
    # Plot per-speaker WERs statistics per model per demographic group
    bias_visualizer.plot_error_rates('Group')
    # Plot per-speaker WERs statistics per model per cluster
    bias_visualizer.plot_error_rates('Cluster')

    # Plot linear correlation coefficients between per-speaker WERs and their extracted feature vector
    bias_visualizer.correlation_features_errors()

    # Return WERs per model per group and per cluster
    return error_rates_per_group, error_rates_per_cluster

def bias_metric_pipeline2(file_path, speaking_style, custom_features, overall_bias_measure: OverallBias,
                          asr_performance_data: AsrPerformanceData, excluded_features=[]):
    """More efficient bias pipeline that re-uses shared data when more feature sets are compared.
    """    
    # Prepare dataframe of extracted features
    data_prep = DataPrep(file_path, speaking_style=speaking_style, 
                         feature_dict=custom_features, dropcolumns=excluded_features)
    data_prep.normalize_data(scaler=MinMaxScaler(), check_normality=False)

    # Initialize clustering algorithm
    clusteringAlgorithm = AgglomerativeClustering(n_clusters=5, linkage='ward')

    # Apply clustering to feature vectors of extracted acoustic and prosodic features
    clustering = Clustering(data_prep, clusteringAlgorithm)
    if clustering.get_group_sizes()[:, 1].min() < 10:
        return None

    # Instantiate class where extracted features and ASR performance are merged
    clustering_performance = GroupingPerformance(clustering, asr_performance_data)

    # Calculate Overall Difference Bias for the ASR models for clusters
    overall_bias = clustering_performance.calculate_bias('Cluster', overall_bias_measure)

    # Return Overall Bias per model
    return overall_bias

def compare_feature_subsets(results):
    subset_visualizer = SubsetsVisualization()
    subset_visualizer.plot_error_description_per_feature_subset(results=results)

def main():
    with open("python\\config\\config.json", 'r') as file:
        config = json.load(file)
    
    features_config = config['feature_extraction']
    speaking_styles, feature_vectors_filepaths = zip(*[
        (entry['speaking_style'], features_config['base_path'] + "\\" + entry['file_name'])
        for entry in features_config['data_sources']])
    
    extracted_features = [
        ("Mean Pitch (Hz)", "Pitch", "Hertz"),
        ("Mean Articulation Rate (phpm)", "Artic. Rate", "Phonemes per Minute"),
        ("Mean Duration E (s)", "Duration /ɛ/", "Seconds"),
        ("Mean Duration A (s)", "Duration /ɑ/", "Seconds"),
        ("Mean Duration u (s)", "Duration /u/", "Seconds"),
        ("Mean Duration O (s)", "Duration /ɔ/", "Seconds"),
        ("Mean Duration @ (s)", "Duration /ə/", "Seconds"),
        ("Mean F1 E (bark)", "F1 /ɛ/", "Bark"),
        ("Mean F2 E (bark)", "F2 /ɛ/", "Bark"),
        ("Mean F1 A (bark)", "F1 /ɑ/", "Bark"),
        ("Mean F2 A (bark)", "F2 /ɑ/", "Bark"),
        ("Mean F1 u (bark)", "F1 /u/", "Bark"),
        ("Mean F2 u (bark)", "F2 /u/", "Bark"),
        ("Mean F1 O (bark)", "F1 /ɔ/", "Bark"),
        ("Mean F2 O (bark)", "F2 /ɔ/", "Bark"),
        ("Mean F1 @ (bark)", "F1 /ə/", "Bark"),
        ("Mean F2 @ (bark)", "F2 /ə/", "Bark"),
    ]

    # Use numbers (that are cut off in plots) to control the order of the plots
    feature_subset_names = [
        "2Artic. Rate",
        "1Pitch",
        "3Adank+",
        "4Feng+",
        "5Full Set",
    ]

    feature_subsets = [
        [ # minimal subset Artic
            "Mean Articulation Rate (phpm)",
        ],
        [ # minimal subset Pitch
            "Mean Pitch (Hz)",
        ],
        [ # Adank+
            "Mean F2 A (bark)",
            "Mean F2 u (bark)",
            "Mean F1 O (bark)",
            "Mean F2 O (bark)",
            "Mean F1 E (bark)",
            "Mean F2 E (bark)",
        ],
        [ # Feng+
            "Mean Pitch (Hz)",
            "Mean Articulation Rate (phpm)",
            "Mean Duration O (s)",
            "Mean Duration @ (s)",
            "Mean F1 O (bark)",
            "Mean F2 O (bark)",
            "Mean F1 @ (bark)",
            "Mean F2 @ (bark)",
        ],
        [ # maximal subset
            "Mean Pitch (Hz)",
            "Mean Articulation Rate (phpm)",
            "Mean Duration E (s)",
            "Mean Duration A (s)",
            "Mean Duration u (s)",
            "Mean Duration O (s)",
            "Mean Duration @ (s)",
            "Mean F1 E (bark)",
            "Mean F2 E (bark)",
            "Mean F1 A (bark)",
            "Mean F2 A (bark)",
            "Mean F1 u (bark)",
            "Mean F2 u (bark)",
            "Mean F1 O (bark)",
            "Mean F2 O (bark)",
            "Mean F1 @ (bark)",
            "Mean F2 @ (bark)"
        ],
    ]

    custom_features_dict = {csvfield: (label, unit) for (csvfield, label, unit) in extracted_features}

    results = {}

    for i, feature_subset in enumerate(feature_subsets):
        # Find list of features to ignore from the input data
        # Warning: any feature in the csv but not in custom_features_dict is not excluded from the clustering
        excluded_features = [feature for feature in custom_features_dict if feature not in feature_subset]

        for speaking_style, filepath in zip(speaking_styles, feature_vectors_filepaths):
            print(f"#### {speaking_style} {feature_subset} ####")
            print(f"#### {feature_subset_names[i]} ####")
            error_rates_per_group, error_rates_per_cluster = bias_metric_pipeline(file_path=filepath, 
                                speaking_style=speaking_style,
                                custom_features=custom_features_dict, 
                                excluded_features=excluded_features)
            
            # Add demographic groups bias as baseline
            if i == 0:
                results[("0Baseline", speaking_style)] = error_rates_per_group

            # Add feature subset performance to results
            results[(f"{feature_subset_names[i]}", speaking_style)] = error_rates_per_cluster

    # Plot the comparison of all feature subsets 
    compare_feature_subsets(results)


if __name__ == "__main__":
    main()
