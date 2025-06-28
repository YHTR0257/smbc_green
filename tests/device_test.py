import sys
import os
import importlib
from typing import Dict, Any, Optional
import warnings

# 警告を抑制（テスト結果を見やすくするため）
warnings.filterwarnings("ignore")


class DeviceDetectionTest:
    """デバイス検出とライブラリテストを行うクラス"""
    
    def __init__(self):
        self.results = {}
        self.device_info = {}
        
    def print_header(self, title: str, char: str = "=", width: int = 60):
        """ヘッダーを印刷"""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)
    
    def print_section(self, title: str, char: str = "-", width: int = 40):
        """セクションヘッダーを印刷"""
        print(f"\n{title}")
        print(char * len(title))
    
    def check_python_environment(self):
        """Python環境の確認"""
        self.print_section("Python環境情報")
        
        python_info = {
            "version": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "architecture": os.uname().machine if hasattr(os, 'uname') else 'Unknown'
        }
        
        for key, value in python_info.items():
            print(f"{key.capitalize()}: {value}")
        
        self.results["python"] = python_info
        return python_info
    
    def check_library_versions(self):
        """ライブラリバージョンの確認"""
        self.print_section("インストール済みライブラリ")
        
        libraries = {
            "torch": "PyTorch",
            "numpy": "NumPy", 
            "pandas": "Pandas",
            "scipy": "SciPy",
            "matplotlib": "Matplotlib",
            "sklearn": "Scikit-learn",
            "lightgbm": "LightGBM",
            "xgboost": "XGBoost",
            "optuna": "Optuna",
            "jupyterlab": "JupyterLab"
        }
        
        library_status = {}
        
        for lib_name, display_name in libraries.items():
            try:
                module = importlib.import_module(lib_name)
                version = getattr(module, "__version__", "Unknown")
                print(f"✅ {display_name}: {version}")
                library_status[lib_name] = {"status": "installed", "version": version}
            except ImportError:
                print(f"❌ {display_name}: Not installed")
                library_status[lib_name] = {"status": "not_installed", "version": None}
        
        self.results["libraries"] = library_status
        return library_status
    
    def detect_pytorch_device(self):
        """PyTorchデバイス検出"""
        self.print_section("PyTorch & CUDA確認")
        
        try:
            import torch
            
            device_info = {
                "pytorch_version": torch.__version__,
                "cuda_compiled": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_count": 0,
                "devices": [],
                "default_device": "cpu"
            }
            
            print(f"PyTorch Version: {device_info['pytorch_version']}")
            print(f"CUDA Compiled Version: {device_info['cuda_compiled']}")
            print(f"CUDA Available: {device_info['cuda_available']}")
            
            if device_info["cuda_available"]:
                device_count = torch.cuda.device_count()
                device_info["device_count"] = device_count
                device_info["default_device"] = "cuda"
                
                print(f"CUDA Devices Count: {device_count}")
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    device_data = {
                        "index": i,
                        "name": props.name,
                        "memory_total": props.total_memory,
                        "memory_total_gb": round(props.total_memory / 1024**3, 1),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multi_processor_count
                    }
                    device_info["devices"].append(device_data)
                    
                    print(f"  GPU {i}: {device_data['name']}")
                    print(f"    Memory: {device_data['memory_total_gb']} GB")
                    print(f"    Compute Capability: {device_data['compute_capability']}")
                
                # cuDNN確認
                if torch.backends.cudnn.is_available():
                    device_info["cudnn_version"] = torch.backends.cudnn.version()
                    device_info["cudnn_enabled"] = torch.backends.cudnn.enabled
                    print(f"cuDNN Version: {device_info['cudnn_version']}")
                    print(f"cuDNN Enabled: {device_info['cudnn_enabled']}")
            else:
                print("ℹ️  CUDA not available - using CPU")
            
            self.device_info = device_info
            self.results["pytorch"] = device_info
            return device_info
            
        except ImportError:
            print("❌ PyTorch not installed")
            return None
    
    def test_pytorch_operations(self):
        """PyTorch基本操作のテスト"""
        self.print_section("PyTorch操作テスト")
        
        try:
            import torch
            
            # デバイス決定
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # テンソル作成と基本操作
            print("\n基本テンソル操作:")
            x = torch.randn(3, 3).to(device)
            y = torch.randn(3, 3).to(device)
            
            print(f"  Tensor creation: ✅ ({x.shape} on {x.device})")
            
            # 行列乗算
            z = torch.mm(x, y)
            print(f"  Matrix multiplication: ✅ ({z.shape} on {z.device})")
            
            # 大きな計算でパフォーマンステスト
            print("\nパフォーマンステスト:")
            large_x = torch.randn(1000, 1000).to(device)
            large_y = torch.randn(1000, 1000).to(device)
            
            import time
            start_time = time.time()
            large_z = torch.mm(large_x, large_y)
            end_time = time.time()
            
            computation_time = end_time - start_time
            print(f"  Large matrix multiplication (1000x1000): ✅")
            print(f"  Computation time: {computation_time:.4f} seconds")
            
            # メモリ使用量（CUDA利用時のみ）
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"  GPU Memory allocated: {memory_allocated:.1f} MB")
                print(f"  GPU Memory reserved: {memory_reserved:.1f} MB")
            
            self.results["pytorch_test"] = {
                "status": "success",
                "device": str(device),
                "computation_time": computation_time
            }
            
            return True
            
        except Exception as e:
            print(f"❌ PyTorch test failed: {e}")
            self.results["pytorch_test"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_ml_libraries(self):
        """機械学習ライブラリのテスト"""
        self.print_section("機械学習ライブラリテスト")
        
        ml_tests = {}
        
        # LightGBM テスト
        try:
            import lightgbm as lgb
            import numpy as np
            
            print("\nLightGBM テスト:")
            X = np.random.rand(100, 5)
            y = np.random.rand(100)
            
            dtrain = lgb.Dataset(X, label=y)
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1
            }
            
            model = lgb.train(params, dtrain, num_boost_round=10, valid_sets=[dtrain], callbacks=[lgb.log_evaluation(0)])
            predictions = model.predict(X[:5])
            
            print(f"  ✅ Training successful")
            print(f"  ✅ Prediction successful (shape: {predictions.shape})")
            
            ml_tests["lightgbm"] = {"status": "success", "version": lgb.__version__}
            
        except Exception as e:
            print(f"  ❌ LightGBM test failed: {e}")
            ml_tests["lightgbm"] = {"status": "failed", "error": str(e)}
        
        self.results["ml_libraries"] = ml_tests
        return ml_tests
    
    def test_data_processing(self):
        """データ処理ライブラリのテスト"""
        self.print_section("データ処理ライブラリテスト")
        
        data_tests = {}
        
        # NumPy テスト
        try:
            import numpy as np
            
            print("NumPy テスト:")
            arr = np.random.randn(1000, 1000)
            result = np.dot(arr, arr.T)
            
            print(f"  ✅ Array creation and computation successful")
            print(f"  ✅ Shape: {result.shape}, dtype: {result.dtype}")
            
            data_tests["numpy"] = {"status": "success", "version": np.__version__}
            
        except Exception as e:
            print(f"  ❌ NumPy test failed: {e}")
            data_tests["numpy"] = {"status": "failed", "error": str(e)}
        
        # Pandas テスト
        try:
            import pandas as pd
            import numpy as np
            
            print("\nPandas テスト:")
            df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
            summary = df.describe()
            
            print(f"  ✅ DataFrame creation successful")
            print(f"  ✅ Statistical summary computed")
            print(f"  ✅ Shape: {df.shape}")
            
            data_tests["pandas"] = {"status": "success", "version": pd.__version__}
            
        except Exception as e:
            print(f"  ❌ Pandas test failed: {e}")
            data_tests["pandas"] = {"status": "failed", "error": str(e)}
        
        self.results["data_processing"] = data_tests
        return data_tests
    
    def generate_summary(self):
        """テスト結果のサマリーを生成"""
        self.print_header("テスト結果サマリー")
        
        # デバイス情報
        if self.device_info:
            if self.device_info.get("cuda_available"):
                device_status = f"✅ GPU ({self.device_info['devices'][0]['name']})"
            else:
                device_status = "ℹ️  CPU Only"
        else:
            device_status = "❌ PyTorch Not Available"
        
        print(f"デバイス: {device_status}")
        
        # ライブラリ状況
        if "libraries" in self.results:
            installed_count = sum(1 for lib in self.results["libraries"].values() 
                                if lib["status"] == "installed")
            total_count = len(self.results["libraries"])
            print(f"ライブラリ: {installed_count}/{total_count} installed")
        
        # テスト結果
        test_categories = ["pytorch_test", "ml_libraries", "data_processing"]
        for category in test_categories:
            if category in self.results:
                category_name = {
                    "pytorch_test": "PyTorch操作",
                    "ml_libraries": "ML Libraries", 
                    "data_processing": "Data Processing"
                }[category]
                
                if category == "pytorch_test":
                    status = "✅" if self.results[category]["status"] == "success" else "❌"
                    print(f"{category_name}: {status}")
                else:
                    success_count = sum(1 for test in self.results[category].values() 
                                      if test["status"] == "success")
                    total_count = len(self.results[category])
                    print(f"{category_name}: {success_count}/{total_count} passed")
        
        print(f"\n{'='*60}")
        print("🎉 環境確認完了!")
        
        return self.results
    
    def run_all_tests(self):
        """全てのテストを実行"""
        self.print_header("デバイス検出・環境確認テスト")
        
        self.check_python_environment()
        self.check_library_versions()
        self.detect_pytorch_device()
        self.test_pytorch_operations()
        self.test_ml_libraries()
        self.test_data_processing()
        self.generate_summary()
        
        return self.results


def main():
    """メイン関数"""
    tester = DeviceDetectionTest()
    results = tester.run_all_tests()
    
    # 結果をJSONで出力したい場合（オプション）
    # import json
    # print("\n" + "="*60)
    # print("JSON Results:")
    # print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()