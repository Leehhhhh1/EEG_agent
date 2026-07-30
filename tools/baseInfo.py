def baseInfo(file_path):
    # 读取文件。
    """处理 base Info 相关逻辑。"""
    with open(file_path, 'rb') as f:
        header_bytes = f.read(256)  # 保留的开发备注。

    # 保留的开发备注。
    if len(header_bytes) < 256:
        raise ValueError("File is less than 256 bytes; not a complete EDF header")

    # 读取 EDF 文件头信息。
    edf_version = header_bytes[0:8].decode('ascii').strip()       # 读取 EDF 文件头信息。
    patient_id = header_bytes[8:88].decode('ascii').strip()       # 解析基础字段。
    recording_id = header_bytes[88:168].decode('ascii').strip()   # 解析基础字段。
    startdate = header_bytes[168:176].decode('ascii').strip()     # 保留的开发备注。
    starttime = header_bytes[176:184].decode('ascii').strip()     # 保留的开发备注。
    header_bytes_length = header_bytes[184:192].decode('ascii').strip()  # 读取 EDF 文件头信息。
    reserved = header_bytes[192:236].decode('ascii').strip()      # 保留的开发备注。
    num_data_records = header_bytes[236:244].decode('ascii').strip()      # 保留的开发备注。
    duration_of_record = header_bytes[244:252].decode('ascii').strip()    # 保留的开发备注。
    num_signals = header_bytes[252:260].decode('ascii').strip()           # 保留的开发备注。

    # 解析基础字段。
    patient_parts = patient_id.split()
    patient_number = patient_parts[0] if len(patient_parts) > 0 else None
    sex = patient_parts[1] if len(patient_parts) > 1 and patient_parts[1] in ['M', 'F'] else None
    # 保留的开发备注。
    age_str = next((p for p in patient_parts if p.startswith("Age:")), None)
    age = int(age_str[4:]) if age_str else None

    # 构建中间结果。
    minimal_info = {
        "patient_id": patient_number,
        "sex": sex,
        "age": age,
        "start_date": startdate,
        "start_time": starttime,
        "data_duration": f"[0 - {int(num_data_records) * float(duration_of_record)}]",  # 保留的开发备注。
    }

    # 解析基础字段。
    minimal_info = {k: v for k, v in minimal_info.items() if v is not None}
    return minimal_info


def get_age_factor(age, age_factors):
    """获取 get age factor 相关信息。"""
    for factor in age_factors:
        if factor["min_age"] <= age < factor["max_age"]:
            return factor
    return None  # 返回值说明。
