settings = {
    'Sasho': {
        'combinedinputfile': 'logs_aggregated_full.csv',
        'anomalyinputfile': 'logs_aggregated_anomalies_only.csv',
        'normalinputfile': 'logs_aggregated_normal_only.csv',
        'inputdir': 'data/openstack/sasho/raw/',
        'parseddir': 'data/openstack/sasho/parsed/',
        'resultsdir': 'data/openstack/sasho/results/bert',
        'embeddingspickledir': 'data/openstack/sasho/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/sasho/embeddings/',
        'logtype': 'OpenStack'
    },
    'Utah': {
        'combinedinputfile': 'openstack_18k_plus_52k',
        'anomalyinputfile': 'openstack_18k_anomalies',
        'normalinputfile': 'openstack_52k_normal',
        'inputdir': 'data/openstack/utah/raw/',
        'parseddir': 'data/openstack/utah/parsed/',
        'resultsdir': 'data/openstack/utah/results/bert',
        'embeddingspickledir': 'data/openstack/utah/padded_embeddings_pickle/',
        'embeddingsdir': 'data/openstack/utah/embeddings/',
        'logtype': 'OpenStack'
    }
}