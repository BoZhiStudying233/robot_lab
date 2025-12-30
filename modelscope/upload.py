from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ms-2324015c-02d2-447b-ac16-5049cafd4bf0'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

from modelscope.hub.constants import Licenses, ModelVisibility

# owner_name = 'bozhi233'
# model_name = 'go2x5'
# model_id = f"{owner_name}/{model_name}"

# api.upload_folder(
#     repo_id=f"{owner_name}/{model_name}",
#     folder_path='/data1/duanzhibo/code/robot_lab/logs/rsl_rl/go2_x5_rough/2025-12-15_08-17-54',
#     commit_message='upload model folder to repo',
# )



owner_name = 'bozhi233'
dataset_name = 'go2x5'

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='/data1/duanzhibo/code/robot_lab/logs/rsl_rl/go2_x5_rough/2025-12-15_08-17-54',
    commit_message='upload dataset folder to repo',
    repo_type = 'dataset'
)