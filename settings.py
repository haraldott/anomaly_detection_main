settings = {
    'Sasho': {
        'combinedinputfile': 'logs_aggregated_full.csv',
        'anomalyinputfile': 'logs_aggregated_anomalies_only.csv',
        'normalinputfile': 'logs_aggregated_normal_only.csv',
        'inputdir': 'data/openstack/sasho/raw/',
        'parseddir': 'data/openstack/sasho/parsed/',
        'resultsdir': 'data/openstack/sasho/results/Sasho/',
        'embeddingspickledir': 'data/openstack/sasho/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/sasho/embeddings/',
        'logtype': 'OpenStackSasho',
        'instance_information_file_normal': None,
        'instance_information_file_anomalies': None
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
        'anomalyinputfile': 'openstack_18k_anomalies_sorted_per_request',
        'normalinputfile': 'openstack_137k_normal_sorted_per_request',
        'inputdir': 'data/openstack/utah/raw/sorted_per_request/',
        'parseddir': 'data/openstack/utah/parsed/',
        'resultsdir': 'data/openstack/utah/results/UtahSorted137/',
        'embeddingspickledir': 'data/openstack/utah/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/utah/embeddings/',
        'logtype': 'OpenStack',
        'instance_information_file_normal': 'data/openstack/utah/raw/sorted_per_request_pickle/'
                                            'openstack_137k_normal_information.pickle',
        'instance_information_file_anomalies': 'data/openstack/utah/raw/sorted_per_request_pickle/'
                                               'openstack_18k_anomalies_information.pickle'
    },
    'UtahSashoTransfer': {
        'combinedinputfile': 'logs_aggregated_full.csv',
        'anomalyinputfile': 'logs_aggregated_anomalies_only.csv',
        'normalinputfile': 'logs_aggregated_normal_only.csv',
        'inputdir': 'data/openstack/sasho/raw/',
        'parseddir': 'data/openstack/sasho/parsed/',
        'resultsdir': 'data/openstack/sasho/results/Sasho/',
        'embeddingspickledir': 'data/openstack/sasho/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/sasho/embeddings/',
        'logtype': 'OpenStackSasho',
        'instance_information_file_normal': None,
        'instance_information_file_anomalies': None,
        'combinedinputfile_transfer': 'openstack_137k_plus_18k_sorted_per_request',
        'anomalyinputfile_transfer': 'openstack_18k_anomalies_sorted_per_request',
        'normalinputfile_transfer': 'openstack_137k_normal_sorted_per_request',
        'inputdir_transfer': 'data/openstack/utah/raw/sorted_per_request/',
        'parseddir_transfer': 'data/openstack/utah/parsed/',
        'resultsdir_transfer': 'data/openstack/utah/results/UtahSashoTransfer/',
        'embeddingspickledir_transfer': 'data/openstack/utah/padded_embeddings_pickle/',
        'embeddingsdir_transfer': 'data/openstack/utah/embeddings/',
        'logtype_transfer': 'OpenStack',
        'instance_information_file_normal_transfer': 'data/openstack/utah/raw/sorted_per_request_pickle/'
                                            'openstack_137k_normal_information.pickle',
        'instance_information_file_anomalies_transfer': 'data/openstack/utah/raw/sorted_per_request_pickle/'
                                               'openstack_18k_anomalies_information.pickle'
    }
}