_register_data = None

def registerData(data):
    """处理 register Data 相关逻辑。"""
    global _register_data
    _register_data = data

def getRegisteredData(s: int = None, e: int = None, config: dict = None):
    """获取 get Registered Data 相关信息。"""
    if _register_data is None:
        raise RuntimeError("The data is not registered, please call register_data to load the data first")
    
    if s is not None and e is not None and config is not None:
        fs = config['fs']
        return _register_data[:, s*fs:e*fs]
    
    return _register_data