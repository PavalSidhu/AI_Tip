import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict

def predict_log_file(log_file_path, model_path):
    def guess_direction(src_port, dst_port):
        common_server_ports = {80, 443, 21, 22, 25, 110, 143, 993, 995}
        if dst_port in common_server_ports:
            return 'outgoing'  # Assuming traffic to common server ports as outgoing
        elif src_port in common_server_ports:
            return 'incoming'  # Incoming if the source port is a common server port
        else:
            return 'undetermined'  # Undetermined if none of the conditions match

    # Load the model
    nb_model = joblib.load(model_path)

    # Load the pcap file
    packets = rdpcap(log_file_path)

    # Initialize dictionaries for tracking the first SYN, SYN-ACK, and ACK packets within each flow
    syn_times, synack_times, ack_times = {}, {}, {}

    # Initialize packet processing
    features = []

    # Process each packet
    for packet in packets:
        if 'IP' in packet and ('TCP' in packet or 'UDP' in packet):
            # Basic packet features
            packet_features = {
                'packet_length': len(packet),
                'src_ip': packet['IP'].src,
                'dst_ip': packet['IP'].dst,
                'sttl': packet['IP'].ttl,
                'protocol_type': packet['IP'].proto,
                'sbytes': len(packet),
                'dttl': packet['IP'].ttl,
                'dbytes': len(packet),
                'Stime': packet.time,
            }
            
            # Handling TCP and UDP
            port_layer = 'TCP' if 'TCP' in packet else 'UDP'
            packet_features['dsport'] = packet[port_layer].dport
            packet_features['src_port'] = packet[port_layer].sport
            flow_id = f"{packet['IP'].src}-{packet['IP'].dst}-{packet[port_layer].sport}-{packet[port_layer].dport}-{packet['IP'].proto}"
            packet_features['flow_id'] = flow_id

            # Inferring the direction based on ports
            direction = guess_direction(packet[port_layer].sport, packet[port_layer].dport)
            packet_features['direction'] = direction

            features.append(packet_features)

    # Convert the features list into a DataFrame
    df = pd.DataFrame(features)
    assert 'flow_id' in df.columns, "DataFrame df does not have 'flow_id' column."
    
    feature_names = df.columns.tolist()

    # Now, calculate the TCP handshake timings for each flow
    tcp_handshake_timings = []
    tcp_handshake_timings.append({
        'flow_id': flow_id,
        'synack': 0,  # Default value or NaN if you prefer
        'ackdat': 0,  # Default value or NaN if you prefer
        'tcprtt': 0   # Default value or NaN if you prefer
    })

    for flow_id, syn_time in syn_times.items():
        if flow_id in synack_times and flow_id in ack_times:
            synack_duration = synack_times[flow_id] - syn_time
            ackdat_duration = ack_times[flow_id] - synack_times[flow_id]
            tcprtt = synack_duration + ackdat_duration
            tcp_handshake_timings.append({
                'flow_id': flow_id,  # This ensures 'flow_id' is correctly used
                'synack': synack_duration,
                'ackdat': ackdat_duration,
                'tcprtt': tcprtt
            })


    # Create DataFrame from TCP handshake timings.
    df_tcp_timings = pd.DataFrame(tcp_handshake_timings)

    # Merge on 'flow_id'.
    df = pd.merge(df, df_tcp_timings, on='flow_id', how='left')

    # Calculate the average packet size for each flow (as a stand-in for smeansz and dmeansz under our assumptions)
    avg_packet_size_per_flow = df.groupby('flow_id')['packet_length'].mean().reset_index(name='avg_packet_size')

    # Duplicate the avg_packet_size for both smeansz and dmeansz
    avg_packet_size_per_flow['smeansz'] = avg_packet_size_per_flow['avg_packet_size']
    avg_packet_size_per_flow['dmeansz'] = avg_packet_size_per_flow['avg_packet_size']

    # Merge the average packet sizes back into the original DataFrame
    df = df.merge(avg_packet_size_per_flow[['flow_id', 'smeansz', 'dmeansz']], on='flow_id', how='left')


    # Assume df['direction'] indicates 'incoming' or 'outgoing'
    flow_bytes = df.groupby(['flow_id', 'direction'])['packet_length'].sum().unstack(fill_value=0).reset_index()

    # Attempt to rename columns based on expected structure
    try:
        # Standard case: 'incoming', 'outgoing' (and possibly 'undetermined') are present
        flow_bytes.columns = ['flow_id', 'total_incoming_bytes', 'total_outgoing_bytes']
        if 'undetermined' in flow_bytes.columns:
            # If there's an 'undetermined' direction, drop it
            flow_bytes = flow_bytes.drop(columns=['undetermined'])
    except ValueError as e:
        # print("Error renaming columns, adjusting dynamically based on available data.")
        # Dynamically adjust based on available direction data
        column_mapping = {'flow_id': 'flow_id'}
        for col in flow_bytes.columns[1:]:  # Skip the first column as it's 'flow_id'
            if 'incoming' in col.lower():
                column_mapping[col] = 'total_incoming_bytes'
            elif 'outgoing' in col.lower():
                column_mapping[col] = 'total_outgoing_bytes'
            else:
                # If neither 'incoming' nor 'outgoing' fits, consider it 'undetermined' and prepare to drop
                column_mapping[col] = 'undetermined'
        flow_bytes.rename(columns=column_mapping, inplace=True)
        if 'undetermined' in flow_bytes.columns:
            flow_bytes = flow_bytes.drop(columns=['undetermined'])

    # Ensure only 'flow_id', 'total_incoming_bytes', and 'total_outgoing_bytes' are present
    expected_cols = ['flow_id', 'total_incoming_bytes', 'total_outgoing_bytes']
    missing_cols = set(expected_cols) - set(flow_bytes.columns)
    # Fill missing columns with zeros
    for col in missing_cols:
        flow_bytes[col] = 0

    # Calculate the duration of each flow
    flow_duration = df.groupby('flow_id')['Stime'].agg(['min', 'max']).reset_index()
    flow_duration['duration'] = flow_duration['max'] - flow_duration['min']

    # Merge the bytes and duration DataFrames
    flow_stats = flow_bytes.merge(flow_duration[['flow_id', 'duration']], on='flow_id', how='left')

    # Ensure numerical data types for safety
    flow_stats['duration'] = flow_stats['duration'].astype(float)
    flow_stats['total_incoming_bytes'] = flow_stats['total_incoming_bytes'].astype(float)
    flow_stats['total_outgoing_bytes'] = flow_stats['total_outgoing_bytes'].astype(float)

    # Recalculate Dload and Sload with explicit handling for zero durations
    flow_stats['Dload'] = flow_stats.apply(lambda x: x['total_incoming_bytes'] / x['duration'] if x['duration'] > 0 else 0, axis=1)
    flow_stats['Sload'] = flow_stats.apply(lambda x: x['total_outgoing_bytes'] / x['duration'] if x['duration'] > 0 else 0, axis=1)

    # Merge back to the main DataFrame if necessary
    df = df.merge(flow_stats[['flow_id', 'Dload', 'Sload']], on='flow_id', how='left')

    # Ensure packets are sorted by flow_id and Stime (timestamp)
    df['Stime'] = df['Stime'].astype(float)

    # Now attempt the sort again
    df = df.sort_values(by=['flow_id', 'Stime'])

    # Calculate inter-packet times within each flow
    df['time_diff'] = df.groupby('flow_id')['Stime'].diff()

    df = df.sort_values(by=['flow_id', 'Stime'])

    # Calculate 'Ltime' as the timestamp of the last packet in each flow.
    # This will replicate the 'Stime' of the last packet in each group (flow) to all packets in that group.
    df['Ltime'] = df.groupby('flow_id')['Stime'].transform('last')

    # Optional: Calculate the duration as the difference between 'Ltime' and 'Stime' if needed
    df['duration'] = df['Ltime'] - df['Stime']

    # Separate calculations for incoming (Dintpkt) and outgoing (Sintpkt) packets
    df_incoming = df[df['direction'] == 'incoming']
    df_outgoing = df[df['direction'] == 'outgoing']

    # Average inter-packet times for Dintpkt and Sintpkt
    avg_dintpkt = df_incoming.groupby('flow_id')['time_diff'].mean().reset_index(name='Dintpkt')
    avg_sintpkt = df_outgoing.groupby('flow_id')['time_diff'].mean().reset_index(name='Sintpkt')

    # Merge the average inter-packet times back to the main DataFrame if needed
    df = df.merge(avg_dintpkt, on='flow_id', how='left')
    df = df.merge(avg_sintpkt, on='flow_id', how='left')

    # Count flows per source service (using src_ip, src_port, and protocol_type to define a service)
    ct_srv_src = df.groupby(['src_ip', 'src_port', 'protocol_type'])['flow_id'].nunique().reset_index(name='ct_srv_src')
    # Create 'src_service' in ct_srv_src for merging
    ct_srv_src['src_service'] = ct_srv_src['src_ip'] + ':' + ct_srv_src['src_port'].astype(str) + ':' + ct_srv_src['protocol_type'].astype(str)

    # Count flows per destination service (using dst_ip, dsport, and protocol_type)
    ct_srv_dst = df.groupby(['dst_ip', 'dsport', 'protocol_type'])['flow_id'].nunique().reset_index(name='ct_srv_dst')
    # Create 'dst_service' in ct_srv_dst for merging
    ct_srv_dst['dst_service'] = ct_srv_dst['dst_ip'] + ':' + ct_srv_dst['dsport'].astype(str) + ':' + ct_srv_dst['protocol_type'].astype(str)

    # Ensure df has 'src_service' and 'dst_service' for merging
    df['src_service'] = df['src_ip'] + ':' + df['src_port'].astype(str) + ':' + df['protocol_type'].astype(str)
    df['dst_service'] = df['dst_ip'] + ':' + df['dsport'].astype(str) + ':' + df['protocol_type'].astype(str)

    # Merge the service count data back into the original DataFrame
    df = df.merge(ct_srv_src[['src_service', 'ct_srv_src']], on='src_service', how='left')
    df = df.merge(ct_srv_dst[['dst_service', 'ct_srv_dst']], on='dst_service', how='left')
    # Count unique TTL values per flow to represent ct_state_ttl
    ct_state_ttl = df.groupby('flow_id')['sttl'].nunique().reset_index(name='ct_state_ttl')

    # Merge the ct_state_ttl metric back into the main DataFrame
    df = df.merge(ct_state_ttl, on='flow_id', how='left')

    # Group by source and destination IP addresses, then count the unique 'flow_id' for each group
    ct_dst_src_ltm = df.groupby(['src_ip', 'dst_ip'])['flow_id'].nunique().reset_index(name='ct_dst_src_ltm')

    # Merge this count back into the main DataFrame
    # To associate the long-term connection count with each flow, we'll merge on 'src_ip' and 'dst_ip'
    df = pd.merge(df, ct_dst_src_ltm, on=['src_ip', 'dst_ip'], how='left')
    
    df_for_prediction = df[['ct_state_ttl', 'sttl', 'tcprtt', 'dsport', 'sbytes', 'synack', 'dttl',
                            'smeansz', 'ackdat', 'ct_srv_dst', 'Dload', 'Sload', 'ct_srv_src', 'dmeansz',
                            'dbytes', 'Stime', 'Ltime', 'ct_dst_src_ltm', 'Dintpkt', 'Sintpkt']].copy()

    df_for_prediction.fillna(0, inplace=True)

    predictions = nb_model.predict(df_for_prediction)

    unique_entries = set()
    detailed_predictions = []
    for i, prediction in enumerate(predictions):
        if prediction != "Generic":  # Skip "Generic" predictions
            detailed_info = {'Prediction': prediction, 'Src IP': df.iloc[i]['src_ip'], 'Dst IP': df.iloc[i]['dst_ip']}
            entry_str = str(detailed_info)  # Use string representation for uniqueness check
            if entry_str not in unique_entries:
                unique_entries.add(entry_str)
                detailed_predictions.append(detailed_info)

    return detailed_predictions

