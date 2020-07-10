import sys
# !{sys.executable} -m pip install plotly
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cosine
from wordembeddings.transform_bert import get_sentence_vectors
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

# sasho_templates_location = '/Users/haraldott/Development/thesis/detector/data/openstack/sasho/parsed/logs_aggregated_normal_only_spr.csv_templates'
#utah_templates_location = '/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/18k_spr_templates'

# sasho = get_sentence_vectors(templates_location=sasho_templates_location)
#utah = get_sentence_vectors(templates_location=utah_templates_location, bert_model='bert-base-uncased')



# sasho_templates_location = '/Users/haraldott/Development/thesis/detector/data/openstack/sasho/parsed/logs_aggregated_normal_only_spr.csv_templates'
utah_templates_location = '/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/18k_spr_templates'

# sasho_finetune = get_sentence_vectors(templates_location=sasho_templates_location,
#                                       bert_model='wordembeddings/finetuning-models/Sasho/')
utah_cleansing = get_sentence_vectors(templates_location=utah_templates_location,
                                      bert_model='finetuning-models/18k_spr_templates_cleansed')



cosine_distances = []

number_of_vals = 0
sum_of_distances = 0
for i, outer_template_vector in enumerate(utah_cleansing):
    temp_cosine_distances = []
    temp_cosine_distances[0:i] = i * [0]
    for inner_ind in range(i, len(utah_cleansing)):
        dist = cosine(outer_template_vector, utah_cleansing[inner_ind])
        temp_cosine_distances.append(dist)
        number_of_vals += 1
        sum_of_distances = sum_of_distances + dist
    cosine_distances.append(temp_cosine_distances)
average_distance = sum_of_distances / number_of_vals
print("average distance:  {}".format(average_distance))

mask = np.zeros_like(cosine_distances)
mask[np.tril_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(cosine_distances, mask=mask, vmax=.6, square=True, cmap="YlOrRd_r")
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    plt.savefig("finetuning.png", dpi=300)
    plt.clf()


# fig = px.imshow(cosine_distances, range_color=[0, 0.6])
# # fig = go.Figure(data=go.Heatmap(z=cosine_distances))
# fig.update_layout(
#     yaxis=dict(
#         tickmode='linear',
#         tick0=0,
#         dtick=1
#     ),
#     xaxis=dict(
#         tickmode='linear',
#         tick0=0,
#         dtick=1
#     ), height=700, width=700
# )
# fig.update_yaxes(autorange=True)
# fig.show()
# fig = go.Figure(data=go.Heatmap(
#                    x=['0: <*> Creating image',
#                       '1: <*> VM <*> (Lifecycle Event)',
#                       '2: <*> During sync_power_state the instance has a pending task (spawning). Skip.',
#                       '3: <*> Instance <*> successfully.',
#                       '4: <*> Took <*>.<*> seconds to <*> the instance on the hypervisor.',
#                       '5: <*> Took <*>.<*> seconds to build instance.',
#                       '6: <*> Terminating instance',
#                       '7: <*> Deleting instance files <*>',
#                       '8: <*> Deletion of <*>complete',
#                       '9: <*> Took <*>.<*> seconds to deallocate network for instance.',
#                       '10: <*> Attempting claim: memory <*> MB, disk <*> GB, vcpus <*> CPU',
#                       '11: <*> Total memory: <*> MB, used: <*>.<*> MB',
#                       '12: <*> memory limit: <*>.<*> MB, free: <*>.<*> MB',
#                       '13: <*> Total disk: <*> GB, used: <*>.<*> GB',
#                       '14: <*> <*> limit not specified, defaulting to unlimited',
#                       '15: <*> Total vcpu: <*> VCPU, used: <*>.<*> VCPU',
#                       '16: <*> Claim successful'],
#                    z=cosine_distances))



cosine_distances_finetune = []

for outer_template_vector in utah_cleansing:
    temp_cosine_distances = []
    for inner_template_vector in utah_cleansing:
        temp_cosine_distances.append(cosine(outer_template_vector, inner_template_vector))
    cosine_distances_finetune.append(temp_cosine_distances)

fig = go.Figure(data=go.Heatmap(
    z=cosine_distances_finetune))
fig.show()



cosine_distances_diff = []
for i, j in zip(cosine_distances_finetune, cosine_distances):
    cos_diff_temp = []
    for k, l in zip(i, j):
        cos_diff_temp.append(abs(k - l))
    cosine_distances_diff.append(cos_diff_temp)

fig = go.Figure(data=go.Heatmap(z=cosine_distances_diff))
fig.show()




