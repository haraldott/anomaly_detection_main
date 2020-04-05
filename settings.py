settings = {
    'Sasho': {
        'combinedinputfile': 'combined',
        'anomalyinputfile': 'logs_aggregated_anomalies_only_spr_corpus',
        'normalinputfile': 'logs_aggregated_normal_only_spr.csv',
        'inputdir': 'data/openstack/sasho/raw/',
        'parseddir': 'data/openstack/sasho/parsed/',
        'resultsdir': 'data/openstack/sasho/results/Sasho/',
        'embeddingspickledir': 'data/openstack/sasho/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/sasho/embeddings/',
        'logtype': 'OpenStackSasho',
        'instance_information_file_normal': 'data/openstack/sasho/raw/sorted_per_request_pickle/logs_aggregated_normal_only_spr.pickle',
        'instance_information_file_anomalies': 'data/openstack/sasho/raw/sorted_per_request_pickle/logs_aggregated_anomalies_only_spr.pickle'
    },
    'Utah': {
        'combinedinputfile': 'openstack_18k_plus_52k',
        'anomalyinputfile': 'openstack_18k_anomalies',
        'normalinputfile': 'openstack_52k_normal',
        'inputdir': 'data/openstack/utah/raw/',
        'parseddir': 'data/openstack/utah/parsed/',
        'resultsdir': 'data/openstack/utah/results/Utah/',
        'embeddingspickledir': 'data/openstack/utah/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/utah/embeddings/',
        'logtype': 'OpenStack',
        'instance_information_file_normal': None,
        'instance_information_file_anomalies': None
    },
    'UtahSorted52': {
        'combinedinputfile': 'openstack_52k_plus_18k_sorted_per_request',
        'anomalyinputfile': 'openstack_18k_anomalies_sorted_per_request',
        'normalinputfile': 'openstack_52k_normal_sorted_per_request',
        'inputdir': 'data/openstack/utah/raw/sorted_per_request/',
        'parseddir': 'data/openstack/utah/parsed/',
        'resultsdir': 'data/openstack/utah/results/UtahSorted137/',
        'embeddingspickledir': 'data/openstack/utah/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/utah/embeddings/',
        'logtype': 'OpenStack',
        'instance_information_file_normal': 'data/openstack/utah/raw/sorted_per_request_pickle/'
                                            'openstack_52k_normal_information.pickle',
        'instance_information_file_anomalies': 'data/openstack/utah/raw/sorted_per_request_pickle/'
                                               'openstack_18k_anomalies_information.pickle'
    },
    'UtahSorted137': {
        'combinedinputfile': 'openstack_137k_plus_18k_sorted_per_request',
        'anomalyinputfile': 'openstack_18k_random_lines_anomalies_new',
        'normalinputfile': '137k_spr',
        'inputdir': 'data/openstack/utah/raw/sorted_per_request/',
        'parseddir': 'data/openstack/utah/parsed/',
        'resultsdir': 'data/openstack/utah/results/UtahSorted137/',
        'embeddingspickledir': 'data/openstack/utah/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/utah/embeddings/',
        'logtype': 'OpenStack',
        'instance_information_file_normal': 'data/openstack/utah/raw/sorted_per_request_pickle/137k_spr.pickle',
        'instance_information_file_anomalies': 'data/openstack/utah/raw/sorted_per_request_pickle/openstack_18k_random_lines_anomalies_new.pickle'
    },

    'Normal': {
            # files
            "raw_normal": "137k_spr",
            "raw_anomaly": "18k_spr",
            # dirs
            "raw_dir": "data/openstack/utah/raw/sorted_per_request/",
            "parsed_dir": "data/openstack/utah/parsed/",
            "results_dir": "data/openstack/utah/results/UtahSorted",
            "embeddings_dir": "data/openstack/utah/embeddings/",
            # logtype for drain parsing
            "logtype": "OpenStack",
            # instance information files
            "instance_information_file_normal": "data/openstack/utah/raw/sorted_per_request_pickle/137k_spr.pickle",
            "instance_information_file_anomalies_pre_inject": "data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle",
            "instance_information_file_anomalies_injected": "data/openstack/utah/raw/sorted_per_request_pickle/anomalies/"
    },

    'UtahSashoTransfer': {
        'dataset_1':
            {
                # files
                "raw_normal": "logs_aggregated_normal_only_spr.csv",
                # dirs
                "raw_dir": "data/openstack/sasho/raw/sorted_per_request/",
                "parsed_dir": "data/openstack/sasho/parsed/",
                "embeddings_dir": "data/openstack/sasho/embeddings/",
                # logtype for drain
                "logtype": "OpenStackSasho",
                # instance information file
                "instance_information_file_normal": "data/openstack/sasho/raw/sorted_per_request_pickle/logs_aggregated_normal_only_spr.pickle",
             },
        'dataset_2':
            {
                # files
                "raw_normal": "137k_spr",
                "raw_anomaly": "18k_spr",
                # dirs
                "raw_dir": "data/openstack/utah/raw/sorted_per_request/",
                "parsed_dir": "data/openstack/utah/parsed/",
                "results_dir": "data/openstack/utah/results/UtahSashoTransfer",
                "embeddings_dir": "data/openstack/utah/embeddings/",
                # logtype for drain parsing
                "logtype": "OpenStack",
                # instance information files
                "instance_information_file_normal": "data/openstack/utah/raw/sorted_per_request_pickle/137k_spr.pickle",
                "instance_information_file_anomalies_pre_inject": "data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle",
                "instance_information_file_anomalies_injected": "data/openstack/utah/raw/sorted_per_request_pickle/anomalies/"
             }

    }
}
