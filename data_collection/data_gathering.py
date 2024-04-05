import pandas as pd

nsl_column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'class', 'label'
]

nb_column_names = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes'
    ,'sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts'
    ,'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len'
    ,'Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat'
    ,'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
    'ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm',
    'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label'
]
def load_nsl_kdd_dataset():
    # Load the NSL-KDD dataset from the CSV file
    dataset_path = '/home/pavalsidhu/AI_TIP/data_storage/KDDTrain+.txt'
    nsl_kdd_df = pd.read_csv(dataset_path, header=None)
    
    # Assign column names
    nsl_kdd_df.columns = nsl_column_names
    
    # Return the loaded DataFrame
    return nsl_kdd_df

def load_nb15_dataset():
    # Load the NB15-BT dataset from the CSV file
    dataset_path = '/home/pavalsidhu/AI_TIP/data_storage/UNSW-NB15_combined.csv'

    nb15_bt_df = pd.read_csv(dataset_path, header=None, low_memory=False)
    
    # Assign column names
    nb15_bt_df.columns = nb_column_names
    
    # print(nb15_bt_df)
    # Return the loaded DataFrame
    return nb15_bt_df
