# API Map

> 自動產生，掃描日期：2026-05-28
> 掃描範圍：`combine_all_new/`
>
> **欄位說明**
>
> - **Parameters**：`name: Type = default`；`self` / `cls` 省略型別。
> - **Return Type**：function 註解的回傳型別；無註解時掃 `return` 推論（純 `return` / 沒 `return` → `None`）；無法推論才標 `Unknown`。
> - **Called By**：直接呼叫此 function 的位置（只追蹤 `combine_all_new/` 內部）。`ClassName()` 實例化視為呼叫 `__init__`；`::<module>` 表示 module 層級程式碼（含 `if __name__ == "__main__":`）。
>
> 缺型別註解時，會依參數名稱推論（包含專案內部 class 如 `osu_interface` / `model_interface` / `RL_interface` / `TCPServer` 等），無法推論才標 `Unknown`。

---

## `combine_all_new/config.py`

### Classes

#### `TAGS(Enum)`

*(Enum 成員：INFO='[INFO]', MARKER='[MARKER]', WARNING='[WARNING]', ERROR='[ERROR]')*

---

## `combine_all_new/core/__init__.py`

*(無 class / function)*

---

## `combine_all_new/core/account_manager.py`

### Classes

#### `AccountManager`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self` | `None` | `core.account_manager.py::<module>`<br>`core.app_controller.MainController.__init__` |
| `load_accounts` | `self` | `list` | `core.account_manager.AccountManager.__init__` |
| `init_account` | `self`, `account_name: str`, `RL_interface: RL_interface`, `model_interface: model_interface` | `bool` | `core.account_manager.AccountManager.login_as` |
| `login_as` | `self`, `account_name: str`, `RL_interface: RL_interface`, `model_interface: model_interface` | `None` | `core.account_manager.AccountManager.run_account_login` |
| `run_account_login` | `self`, `osu_interface: osu_interface`, `RL_interface: RL_interface`, `model_interface: model_interface` | `str` | `core.app_controller.MainController.before_start` |

---

## `combine_all_new/core/app_controller.py`

### Classes

#### `MainController`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self` | `None` | `main.main` |
| `initialize_lsl` | `self` | `setup_lsl_inlet \| None` | `core.app_controller.MainController.__init__` |
| `on_tcp_message` | `self`, `msg: str`, `server: TCPServer` | `None` | — |
| `start` | `self` | `None` | `core.app_controller.MainController.before_start` |
| `_run_model_select` | `self` | `None` | `core.app_controller.MainController.start` |
| `before_start` | `self` | `None` | `main.main` |

---

## `combine_all_new/core/intermission.py`

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `run_intermission` | `osu_interface: osu_interface`, `model_interface: model_interface`, `RL_interface: RL_interface`, `map_notes: list[dict]` | `str` | `core.app_controller.MainController.start` |
| `_run_model_select` | `osu_interface: osu_interface`, `model_interface: model_interface` | `None` | `core.intermission.run_intermission` |

---

## `combine_all_new/core/level_runner.py`

### Classes

#### `LevelRunner`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `osu_interface: osu_interface`, `model_interface: model_interface`, `RL_interface: RL_interface` | `None` | `core.app_controller.MainController.__init__` |
| `run_pause_menu` | `self` | `str` | `core.level_runner.LevelRunner.run_level`<br>`core.level_runner.LevelRunner.run_zero_level` |
| `run_level` | `self`, `map_file: str`, `tcp_server: TCPServer`, `rl_loop_worker_fn: Callable` | `tuple` | `core.app_controller.MainController.start` |
| `run_zero_level` | `self`, `map_file: str`, `tcp_server: TCPServer` | `tuple` | `core.level_runner.LevelRunner.run_level` |

---

## `combine_all_new/core/rl_loop.py`

### Classes

#### `RLLoopWorker`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `model_interface: model_interface`, `RL_interface: RL_interface`, `stableDataDir: str` | `None` | `core.app_controller.MainController.__init__`<br>`test_rl_all.TestRLLoopWorker.setUp` |
| `load_hq_buffer` | `self`, `hqBufferPath: str` | `None` | `core.app_controller.MainController.before_start` |
| `save_hq_buffer` | `self` | `None` | `core.app_controller.MainController.start` |
| `start` | `self` | `None` | `core.app_controller.MainController.start` |
| `stop` | `self` | `None` | `core.app_controller.MainController.start` |
| `_snapshot_realtime_data` | `self` | `tuple` | `core.rl_loop.RLLoopWorker.worker`<br>`test_rl_all.TestRLLoopWorker.test_bandit_save_after_update`<br>`test_rl_all.TestRLLoopWorker.test_context_and_action`<br>`test_rl_all.TestRLLoopWorker.test_phase2_reward_and_update`<br>`test_rl_all.TestRLLoopWorker.test_snapshot_shape` |
| `worker` | `self` | `None` | — |

---

## `combine_all_new/core/test_model_runner.py`

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `run_test_model` | `osu_interface: osu_interface`, `model_interface: model_interface` | `None` | `core.app_controller.MainController.start` |

---

## `combine_all_new/eeg/__init__.py`

*(無 class / function)*

---

## `combine_all_new/eeg/braindecode_trainer.py`

### Classes

#### `BraindecodeTrainer`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `model_class: type`, `model_kwargs: dict`, `dataset: TensorDataset`, `val_dataset: TensorDataset`, `batch_size: int`, `num_epochs: int`, `lr: float`, `ft: bool`, `tcp_client: TCPClient \| None` | `None` | `eeg.braindecode_trainer.py::<module>`<br>`eeg.eeg_trainer.EEGTrainingPipeline._run_braindecode_training`<br>`rl_pretrain.RLPretrainer.run` |
| `_load_data` | `self` | `None` | `eeg.braindecode_trainer.BraindecodeTrainer.__init__` |
| `_init_model` | `self` | `None` | `eeg.braindecode_trainer.BraindecodeTrainer.__init__` |
| `load_checkpoint` | `self`, `checkpoint_path: str` | `None` | `eeg.eeg_trainer.EEGTrainingPipeline._run_braindecode_training`<br>`rl_pretrain.RLPretrainer.run` |
| `_evaluate` | `self`, `loader: DataLoader` | `tuple` | `eeg.braindecode_trainer.BraindecodeTrainer.train` |
| `train` | `self`, `freeze_layer: bool = False`, `patience: int = 10` | `dict` | `eeg.eeg_trainer.EEGTrainingPipeline._run_braindecode_training`<br>`rl_pretrain.RLPretrainer.run` |

---

## `combine_all_new/eeg/config.py`

*(無 class / function)*

---

## `combine_all_new/eeg/data_loader.py`

### Classes

#### `EEGDataLoader_v2`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `valid_ratio: float = 0.25` | `None` | `eeg.data_loader.py::<module>`<br>`eeg.model_interface.model_interface.__init__`<br>`rl_pretrain.RLPretrainer.__init__` |
| `update_dir_path` | `self` | `None` | `eeg.data_loader.EEGDataLoader_v2.setup_the_data_path`<br>`eeg.model_interface.model_interface.rerun_inference_on_pt_files` |
| `update_csv_file_paths` | `self`, `mode: int = 0` | `None` | `eeg.data_loader.EEGDataLoader_v2.setup_the_data_path` |
| `update_pt_file_paths` | `self`, `mode: int = 0` | `None` | `eeg.data_loader.EEGDataLoader_v2.setup_the_data_path`<br>`eeg.model_interface.model_interface.rerun_inference_on_pt_files` |
| `_csv_to_np` | `self`, `csv_path: str` | `dict` | `eeg.data_loader.EEGDataLoader_v2._load_data_from_csv_to_pt` |
| `_cut_trail` | `self`, `DATA_np_dict: dict` | `dict` | `eeg.data_loader.EEGDataLoader_v2._load_data_from_csv_to_pt` |
| `_save_as_pt` | `self`, `DATA_np_dict: dict`, `save_path: str` | `None` | `eeg.data_loader.EEGDataLoader_v2._load_data_from_csv_to_pt` |
| `_load_data_from_csv_to_pt` | `self`, `csv_path: str` | `None` | `eeg.data_loader.EEGDataLoader_v2.setup_the_data_path` |
| `_train_valid_idx` | `self`, `x_np_data: np.ndarray`, `y_np_data: np.ndarray` | `set` | `eeg.data_loader.EEGDataLoader_v2.collect_rl_decisions`<br>`eeg.data_loader.EEGDataLoader_v2.prepare_datasets_RL_v1`<br>`eeg.data_loader.EEGDataLoader_v2.prepare_datasets_v2` |
| `_sliding_window` | `self`, `stride: int`, `segment: np.ndarray` | `list \| None` | `eeg.data_loader.EEGDataLoader_v2.prepare_datasets_RL_v1`<br>`eeg.data_loader.EEGDataLoader_v2.prepare_datasets_RL_v2`<br>`eeg.data_loader.EEGDataLoader_v2.prepare_datasets_v2`<br>`rl_pretrain.RLPretrainer.run` |
| `_to_dataset` | `self`, `segment_list: list`, `label_list: list` | `TensorDataset` | `eeg.data_loader.EEGDataLoader_v2.prepare_datasets_RL_v1`<br>`eeg.data_loader.EEGDataLoader_v2.prepare_datasets_RL_v2`<br>`eeg.data_loader.EEGDataLoader_v2.prepare_datasets_v2`<br>`rl_pretrain.RLPretrainer.run` |
| `prepare_datasets_v2` | `self`, `data_x_list: list`, `data_y_list: list` | `bool \| tuple` | `eeg.data_loader.EEGDataLoader_v2.load_data_and_prepare_dataset` |
| `prepare_datasets_RL_v1` | `self`, `data_x_list: list`, `data_y_list: list`, `probs_list: list`, `RL_interface: RL_interface`, `session_records: list` | `bool \| tuple` | — |
| `collect_rl_decisions` | `self`, `data_x_list: list`, `data_y_list: list`, `probs_list: list`, `RL_interface: RL_interface` | `tuple` | `eeg.data_loader.EEGDataLoader_v2.load_data_and_prepare_dataset`<br>`eeg.model_interface.model_interface.collect_rl_data` |
| `prepare_datasets_RL_v2` | `self`, `session_records: list`, `replay_segments: np.ndarray`, `replay_labels: np.ndarray` | `tuple` | `eeg.data_loader.EEGDataLoader_v2.load_data_and_prepare_dataset`<br>`eeg.model_interface.model_interface.train_rl_loop_update` |
| `setup_the_data_path` | `self`, `mode: int = 0` | `None` | `eeg.model_interface.model_interface.train_or_calib` |
| `load_data_and_prepare_dataset` | `self`, `RL_open: bool = False`, `RL_interface: RL_interface = None`, `replay_segments: np.ndarray = None`, `replay_labels: np.ndarray = None` | `tuple` | `eeg.model_interface.model_interface.train_or_calib` |

---

## `combine_all_new/eeg/eeg_model.py`

### Classes

#### `EEGNet_our_ver(nn.Module)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `n_chans: int = None`, `n_outputs: int = None`, `n_times: int = None`, `F1: int = 4`, `D: int = 2`, `F2: int = 8`, `T_kernSizes: list = [32, 64, 128, 200, 256, 512]`, `S_kernSizes: list = []`, `dropout: float = 0.55` | `None` | `eeg.eeg_model.py::<module>` |
| `_get_feature_dim` | `self`, `n_chans: int`, `n_times: int` | `int` | `eeg.eeg_model.EEGNet_our_ver.__init__` |
| `forward` | `self`, `x: int` | `torch.Tensor` | — |

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `load_shallowfbcsp_params` | `dataset: TensorDataset` | `dict` | `eeg.eeg_trainer.EEGTrainingPipeline.run_training` |
| `load_EEGNet_our_ver_params` | `dataset: TensorDataset` | `dict` | `eeg.eeg_trainer.EEGTrainingPipeline.run_training` |

---

## `combine_all_new/eeg/eeg_predictor.py`

### Classes

#### `EEGPredictor`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `tcp_client: TCPClient \| None` | `None` | `eeg.model_interface.model_interface.__init__` |
| `start_predict_eeg_thread` | `self` | `None` | `eeg.model_interface.model_interface.start_pred_eeg` |
| `end_predict_eeg_thread` | `self` | `None` | `eeg.model_interface.model_interface.end_pred_eeg` |
| `init_model` | `self` | `Unknown` | `eeg.eeg_predictor.EEGPredictor.reload_model`<br>`eeg.eeg_predictor.EEGPredictor.start_predict_eeg_thread`<br>`eeg.eeg_predictor.EEGPredictor.update_model` |
| `update_check_point` | `self`, `path: str` | `None` | — |
| `update_model` | `self` | `None` | — |
| `reload_model` | `self`, `checkpointPath: str` | `None` | — |
| `to_dataset` | `self`, `segment_list: list`, `label_list: list` | `TensorDataset` | `eeg.eeg_predictor.EEGPredictor.set_model_arg` |
| `set_model_arg` | `self` | `Unknown` | `eeg.eeg_predictor.EEGPredictor.__init__` |
| `predict_loop` | `self` | `None` | — |

---

## `combine_all_new/eeg/eeg_reader.py`

### Classes

#### `EEGReader`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `inlet: StreamInlet \| None = None` | `None` | `eeg.eeg_reader.py::<module>`<br>`eeg.model_interface.model_interface.__init__` |
| `start_read_eeg_thread` | `self` | `None` | `eeg.model_interface.model_interface.__init__` |
| `end_read_eeg_thread` | `self` | `None` | — |
| `build_csv_file` | `self` | `None` | `eeg.model_interface.model_interface.start_read_eeg` |
| `close_csv_file` | `self` | `None` | `eeg.model_interface.model_interface.end_read_eeg` |
| `filter_channels` | `self`, `sample_arr: np.ndarray` | `np.ndarray` | `eeg.eeg_reader.EEGReader.update_predict_buffer` |
| `update_predict_buffer` | `self` | `None` | — |

---

## `combine_all_new/eeg/eeg_trainer.py`

### Classes

#### `EEGTrainingPipeline`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `epochs: int = 150`, `freeze_layers: bool = False`, `ft: bool = False`, `tcp_client: TCPClient \| None = None` | `None` | `eeg.eeg_trainer.py::<module>`<br>`eeg.model_interface.model_interface.__init__` |
| `_run_braindecode_training` | `self`, `model_class: type`, `model_kwargs: dict`, `dataset: TensorDataset`, `dataset_valid: TensorDataset`, `save_path: str = None` | `tuple` | `eeg.eeg_trainer.EEGTrainingPipeline.run_training` |
| `run_training` | `self`, `dataset: TensorDataset`, `dataset_val: TensorDataset`, `save_path: str`, `model_class: type = 'EEGNet_our_ver'` | `tuple` | `eeg.model_interface.model_interface.train_or_calib`<br>`eeg.model_interface.model_interface.train_rl_loop_update` |

---

## `combine_all_new/eeg/global_state.py`

*(無 class / function)*

---

## `combine_all_new/eeg/helpers.py`

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `rename_file_with_time` | `name: str` | `str` | `eeg.eeg_reader.EEGReader.build_csv_file`<br>`eeg.helpers.py::<module>` |
| `get_next_version_path` | `base_path: str` | `str` | — |

---

## `combine_all_new/eeg/lsl_utils.py`

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `setup_lsl_inlet` | `stream_name: str = config.RECEIVE_CYGNUS_LSL_STREAM`, `timeout: float = 5.0` | `StreamInlet` | `core.app_controller.MainController.initialize_lsl` |

---

## `combine_all_new/eeg/model_interface.py`

### Classes

#### `model_interface`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `inlet: StreamInlet \| None = None`, `valid_ratio: float = 0.25` | `None` | `core.app_controller.MainController.__init__` |
| `version_setting` | `self`, `version: int` | `None` | — |
| `start_read_eeg` | `self` | `None` | — |
| `start_pred_eeg` | `self` | `None` | — |
| `end_pred_eeg` | `self` | `None` | — |
| `end_read_eeg` | `self` | `None` | — |
| `fetch_raw_eeg` | `self` | `None` | — |
| `write_csv` | `self`, `row: list` | `None` | — |
| `write_csv_batch` | `self`, `rows: list[list]` | `None` | — |
| `train_or_calib` | `self`, `mode: int`, `RL_open: bool`, `RL_interface: RL_interface`, `RL_LOOP_USE: bool = False`, `update_times: int = None` | `bool \| Unknown` | — |
| `collect_rl_data` | `self`, `RL_interface: RL_interface` | `tuple` | — |
| `train_rl_loop_update` | `self`, `session_records: list`, `replay_segments: np.ndarray`, `replay_labels: np.ndarray`, `update_times: int` | `tuple` | — |
| `rerun_inference_on_pt_files` | `self` | `None` | `eeg.model_interface.model_interface.train_or_calib` |
| `get_model_list` | `self` | `list` | `core.app_controller.MainController._run_model_select`<br>`core.app_controller.MainController.before_start`<br>`eeg.model_interface.model_interface.__init__` |
| `set_active_model` | `self`, `idx: int` | `None` | `core.app_controller.MainController._run_model_select`<br>`core.app_controller.MainController.before_start` |

---

## `combine_all_new/eeg/preprocess.py`

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `detrend_epoch` | `epoch: np.ndarray` | `np.ndarray` | — |
| `each_channel_demean` | `data: np.ndarray`, `axis: int = 0` | `np.ndarray` | `core.rl_loop.RLLoopWorker._snapshot_realtime_data`<br>`eeg.data_loader.EEGDataLoader_v2._sliding_window`<br>`eeg.eeg_predictor.EEGPredictor.predict_loop`<br>`eeg.preprocess.py::<module>` |
| `bandpass` | `data: np.ndarray`, `axis: int = 0`, `fs: int = config.SAMPLE_RATE`, `low: float = config.band_pass_low`, `high: float = config.band_pass_high` | `np.ndarray` | `core.rl_loop.RLLoopWorker._snapshot_realtime_data`<br>`eeg.data_loader.EEGDataLoader_v2._sliding_window`<br>`eeg.eeg_predictor.EEGPredictor.predict_loop`<br>`eeg.preprocess.py::<module>` |
| `down_sample` | `data: np.ndarray`, `new_fs: int = config.SAMPLE_RATE` | `np.ndarray` | — |

---

## `combine_all_new/eeg/tcp_utils.py`

### Classes

#### `TCPServer`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `host: str = '0.0.0.0'`, `port: int = 50007`, `on_message: Callable \| None = None` | `None` | `core.app_controller.MainController.__init__` |
| `start` | `self` | `None` | `eeg.tcp_utils.TCPServer.__init__` |
| `stop` | `self` | `None` | — |
| `_run_server` | `self` | `None` | — |
| `_handle_client` | `self`, `conn: socket.socket`, `addr: tuple` | `None` | — |
| `broadcast` | `self`, `message: str` | `None` | — |

#### `TCPClient`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `host: str = '127.0.0.1'`, `port: int = 50007`, `on_message: Callable \| None = None` | `None` | `eeg.model_interface.model_interface.__init__` |
| `start` | `self` | `None` | `eeg.tcp_utils.TCPClient.__init__` |
| `_run_client` | `self` | `None` | — |
| `stop` | `self` | `None` | — |
| `send_messages` | `self`, `msg: str` | `None` | — |

---

## `combine_all_new/global_state.py`

*(無 class / function)*

---

## `combine_all_new/main.py`

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `main` | — | `None` | `main.py::<module>` |

---

## `combine_all_new/osu/__init__.py`

*(無 class / function)*

---

## `combine_all_new/osu/game_view.py`

### Classes

#### `GameView`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `size: int = 800` | `None` | `osu.osu_interface.osu_interface.__init__` |
| `recalc_geometry` | `self` | `None` | `osu.game_view.GameView.__init__`<br>`osu.game_view.GameView.resize` |
| `resize` | `self`, `new_size: int` | `None` | `osu.osu_interface.osu_interface.run_settings_menu` |
| `update_display_mode` | `self`, `fullscreen: bool` | `None` | `osu.osu_interface.osu_interface.__init__`<br>`osu.osu_interface.osu_interface.run_settings_menu` |
| `trigger_effect` | `self`, `lane: int` | `None` | `osu.osu_interface.osu_interface.check_hit` |
| `truncate_text` | `self`, `text: str`, `max_width: int` | `str` | `osu.game_view.GameView.draw_song_select` |
| `get_difficulty_color` | `self`, `note_count: int` | `tuple` | `osu.game_view.GameView.draw_song_select` |
| `draw_arrow` | `self`, `surface: pygame.Surface`, `color: tuple[int, int, int]`, `x: int`, `y: int`, `direction: int`, `size: int = 40`, `filled: bool = True` | `None` | `osu.game_view.GameView.draw_play_screen` |
| `draw_button` | `self`, `text: str`, `x: int`, `y: int`, `w: int`, `h: int`, `active: bool = False` | `pygame.Rect` | `osu.game_view.GameView.draw_main_menu`<br>`osu.game_view.GameView.draw_settings`<br>`osu.game_view.GameView.draw_song_select` |
| `draw_account_select` | `self`, `account_list: list[str]`, `idx: int` | `list` | — |
| `draw_main_menu` | `self` | `dict` | `osu.osu_interface.osu_interface.run_main_menu` |
| `draw_settings` | `self`, `settings: dict` | `dict` | `osu.osu_interface.osu_interface.run_settings_menu` |
| `draw_offset_wizard` | `self`, `current_offset: int`, `avg_diff: float`, `beat_progress: float`, `history: list` | `None` | `osu.osu_interface.osu_interface.run_offset_wizard` |
| `draw_song_select` | `self`, `song_list: list[dict]`, `idx: int` | `tuple` | `osu.osu_interface.osu_interface.run_song_select` |
| `draw_model_select` | `self`, `model_list: list[str]`, `idx: int` | `list` | — |
| `draw_play_screen` | `self`, `notes: list[dict]`, `time_ms: float`, `score: int`, `model_info: dict`, `speed: float`, `accuracy: float` | `None` | — |
| `draw_pause_menu` | `self`, `opt: int` | `list` | — |
| `draw_intermission` | `self`, `correct: int`, `count: int`, `each_count: dict`, `each_count_acc: dict`, `model_name: str` | `None` | — |
| `draw_test_model` | `self`, `model_name: str`, `prediction: int = -1`, `probability: np.ndarray \| None = None`, `prediction_count: int = 0` | `None` | — |
| `draw_loading` | `self`, `msg: str` | `None` | — |
| `close` | `self` | `None` | `osu.osu_interface.osu_interface.run_main_menu`<br>`osu.osu_interface.osu_interface.run_offset_wizard`<br>`osu.osu_interface.osu_interface.run_settings_menu`<br>`osu.osu_interface.osu_interface.run_song_select` |

---

## `combine_all_new/osu/osu_interface.py`

### Classes

#### `osu_interface`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self` | `None` | `core.app_controller.MainController.__init__` |
| `load_settings` | `self` | `dict` | `osu.osu_interface.osu_interface.__init__` |
| `save_settings` | `self` | `None` | `osu.osu_interface.osu_interface.run_offset_wizard`<br>`osu.osu_interface.osu_interface.run_settings_menu` |
| `load_hit_sound` | `self` | `pygame.mixer.Sound \| None` | `osu.osu_interface.osu_interface.__init__` |
| `import_osz_files` | `self` | `None` | `osu.osu_interface.osu_interface.__init__` |
| `load_songs` | `self` | `list` | `osu.osu_interface.osu_interface.__init__` |
| `play_music` | `self`, `song_meta: dict`, `loop: bool = False` | `None` | `osu.osu_interface.osu_interface.run_song_select` |
| `run_main_menu` | `self` | `str` | `core.app_controller.MainController.start` |
| `run_settings_menu` | `self` | `None` | `core.app_controller.MainController.start` |
| `run_offset_wizard` | `self` | `None` | `osu.osu_interface.osu_interface.run_settings_menu` |
| `run_song_select` | `self` | `str \| Unknown` | `core.app_controller.MainController.start` |
| `check_hit` | `self`, `map_notes: list[dict]`, `current_note_index: int`, `key: int`, `game_time_ms: float`, `score: int`, `hits: int` | `tuple` | — |
| `calculate_marker` | `self`, `map_notes: list[dict]`, `current_note_index: int`, `game_time_ms: float` | `tuple` | — |
| `compute_notes` | `self`, `map_notes: list[dict]` | `tuple` | — |

---

## `combine_all_new/osu/osu_parser.py`

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `parse_metadata` | `filepath: str` | `dict` | `osu.osu_interface.osu_interface.load_songs`<br>`osu.osu_parser.parse_osu_map` |
| `parse_osu_map` | `filepath: str`, `speed: float = 0.8` | `dict` | `core.level_runner.LevelRunner.run_level`<br>`core.level_runner.LevelRunner.run_zero_level` |

---

## `combine_all_new/rl/__init__.py`

*(無 class / function)*

---

## `combine_all_new/rl/bandit.py`

### Classes

#### `BanditAction(IntEnum)`

*(Enum 成員：INCLUDE=0, HIGH_WEIGHT=1, LOW_WEIGHT=2, n_actions=3)*

#### `ContextIDX(IntEnum)`

*(Enum 成員：MEAN_ENTROPY=0, MEAN_MARGIN=1, ALPHA_POWER_NORM=2, context_dim=3)*

#### `RewardMLP(nn.Module)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `contextDim: int`, `dropout: float = 0.2` | `None` | — |
| `forward` | `self`, `contextTensor: torch.Tensor` | `torch.Tensor` | — |

#### `NeuralContextualBandit`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `nActions: int = BanditAction.n_actions`, `contextDim: int = ContextIDX.context_dim`, `epsilon: float = 0.5`, `lambdaCost: float = 0.1`, `mu: float = 0.3`, `alp_ent_hyp: float = 1.0`, `weightEntropy: float = 0.5`, `rewardClipMin: float = -2.0`, `rewardClipMax: float = 2.0`, `replayBufferSize: int = 500`, `mlpLearningRate: float = 0.001`, `miniBatchSize: int = 32`, `updateEpochs: int = 5` | `None` | `rl.rl_interface.RL_interface.__init__`<br>`rl_pretrain.RLPretrainer.__init__`<br>`rl_pretrain_premium.PremiumRLPretrainer.__init__`<br>`test_rl_all.TestBanditCore.setUp`<br>`test_rl_all.TestBanditCore.test_pretrained_policy_clean_signal`<br>`test_rl_all.TestBanditCore.test_pretrained_policy_noisy_signal`<br>`test_rl_all.TestBanditCore.test_save_load_roundtrip`<br>`test_rl_all.TestIntermissionCalib.test_bandit_save_after_calib`<br>`test_rl_all.TestRLPretrainSimulator.test_pretrained_model_exists_and_loadable`<br>`test_rl_all.TestRLPretrainSimulator.test_reward_ordering_clean_signal`<br>`test_rl_all.TestRLPretrainSimulator.test_reward_ordering_noisy_signal` |
| `_rule_based_override` | `self`, `context: np.ndarray` | `BanditAction \| None` | `rl.bandit.NeuralContextualBandit.select_action` |
| `_compute_exploration_bonus` | `self`, `actionIdx: int`, `contextTensor: torch.Tensor` | `float` | `rl.bandit.NeuralContextualBandit.select_action` |
| `select_action` | `self`, `context: np.ndarray` | `BanditAction` | `test_rl_all.TestBanditCore.test_pretrained_policy_clean_signal`<br>`test_rl_all.TestBanditCore.test_pretrained_policy_noisy_signal`<br>`test_rl_all.TestBanditCore.test_select_action_valid` |
| `update` | `self`, `context: np.ndarray`, `action: BanditAction`, `reward: float` | `None` | `test_rl_all.TestBanditCore.test_save_load_roundtrip`<br>`test_rl_all.TestBanditCore.test_update_fills_buffer` |
| `update_batch` | `self`, `contexts: list[np.ndarray]`, `actions: list[BanditAction]`, `rewards: list[float]` | `None` | — |
| `_train_mlp` | `self`, `actionIdx: int` | `None` | `rl.bandit.NeuralContextualBandit.update`<br>`rl.bandit.NeuralContextualBandit.update_batch` |
| `_compute_lqp_base` | `self`, `meanEntropy: float` | `float` | `rl.bandit.NeuralContextualBandit.compute_reward` |
| `_compute_gamma` | `self`, `alphaNorm: float`, `meanEntropy: float` | `float` | `rl.bandit.NeuralContextualBandit.compute_reward` |
| `compute_reward` | `self`, `deltaLoss: float`, `meanEntropy: float`, `alphaNorm: float`, `timeCost: float = 1.0` | `float` | `test_rl_all.TestBanditCore.test_compute_reward_clipped`<br>`test_rl_all.TestRLPretrainSimulator.test_reward_ordering_clean_signal`<br>`test_rl_all.TestRLPretrainSimulator.test_reward_ordering_noisy_signal` |
| `save` | `self`, `path: str` | `None` | `test_rl_all.TestBanditCore.test_save_load_roundtrip` |
| `load` | `self`, `path: str` | `None` | `test_rl_all.TestBanditCore.test_pretrained_policy_clean_signal`<br>`test_rl_all.TestBanditCore.test_pretrained_policy_noisy_signal`<br>`test_rl_all.TestBanditCore.test_save_load_roundtrip`<br>`test_rl_all.TestIntermissionCalib.test_bandit_save_after_calib`<br>`test_rl_all.TestRLPretrainSimulator.test_pretrained_model_exists_and_loadable` |

#### `UpdateScheduler`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `scoreThreshold: float = 7.0` | `None` | `rl.rl_interface.RL_interface.__init__`<br>`rl_pretrain.RLPretrainer.__init__`<br>`test_rl_all.TestUpdateScheduler.test_reset_clears_score`<br>`test_rl_all.TestUpdateScheduler.test_trigger_on_threshold` |
| `record` | `self`, `action: BanditAction` | `bool` | `test_rl_all.TestUpdateScheduler.test_reset_clears_score`<br>`test_rl_all.TestUpdateScheduler.test_trigger_on_threshold` |
| `reset` | `self` | `None` | `rl_pretrain.RLPretrainer.run`<br>`test_rl_all.TestUpdateScheduler.test_reset_clears_score` |

#### `ContextNormalizer`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `alpha: float = 0.05` | `None` | `rl.rl_interface.RL_interface.__init__`<br>`rl_pretrain.RLPretrainer.__init__`<br>`test_rl_all.TestContextNormalizer.test_output_in_unit_range`<br>`test_rl_all.TestContextNormalizer.test_repeated_same_input_converges` |
| `update_and_normalize` | `self`, `raw: np.ndarray` | `np.ndarray` | `rl_pretrain.RLPretrainer.run`<br>`test_rl_all.TestContextNormalizer.test_output_in_unit_range`<br>`test_rl_all.TestContextNormalizer.test_repeated_same_input_converges` |

#### `ExperienceReplayMemory`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `stableDataDir: str`, `maxDynamicSize: int = 2000` | `None` | `core.rl_loop.RLLoopWorker.__init__`<br>`test_rl_all.TestExperienceReplayMemory.setUp`<br>`test_rl_all.TestExperienceReplayMemory.test_save_load_buffer` |
| `_load_stable_data` | `self`, `dataDir: str` | `None` | `rl.bandit.ExperienceReplayMemory.__init__` |
| `add_dynamic` | `self`, `segment: np.ndarray`, `label: int`, `action: 'BanditAction'` | `None` | `rl.bandit.ExperienceReplayMemory.add_dynamic_batch`<br>`test_rl_all.TestExperienceReplayMemory.test_add_and_sample_dynamic`<br>`test_rl_all.TestExperienceReplayMemory.test_dynamic_size_limit`<br>`test_rl_all.TestExperienceReplayMemory.test_save_load_buffer` |
| `add_dynamic_batch` | `self`, `segments: list`, `labels: list`, `actions: list` | `None` | `core.rl_loop.RLLoopWorker.worker` |
| `sample` | `self`, `totalCount: int`, `stableRatio: float = 0.3` | `tuple` | `core.rl_loop.RLLoopWorker.worker`<br>`test_rl_all.TestExperienceReplayMemory.test_add_and_sample_dynamic` |
| `save_buffer` | `self`, `savePath: str`, `topK: int = 100` | `None` | `core.rl_loop.RLLoopWorker.save_hq_buffer`<br>`test_rl_all.TestExperienceReplayMemory.test_save_load_buffer` |
| `load_buffer` | `self`, `loadPath: str` | `None` | `core.rl_loop.RLLoopWorker.load_hq_buffer`<br>`test_rl_all.TestExperienceReplayMemory.test_save_load_buffer` |
| `dynamicSize` | `self` | `int` | — |
| `stableSize` | `self` | `int` | — |

---

## `combine_all_new/rl/rl_interface.py`

### Classes

#### `RL_interface`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `bandit: NeuralContextualBandit \| None = None` | `None` | `core.app_controller.MainController.__init__`<br>`rl.rl_interface.py::<module>`<br>`rl_pretrain.RLPretrainer.__init__`<br>`rl_pretrain_premium.PremiumRLPretrainer.__init__`<br>`test_rl_all.make_rl` |
| `_get_band_power` | `self`, `freqs: np.ndarray`, `psd: np.ndarray`, `low: float`, `high: float` | `float \| np.ndarray` | `rl.rl_interface.RL_interface._get_alpha_ratio` |
| `_get_alpha_ratio` | `self`, `eeg_data: np.ndarray` | `float \| np.ndarray` | `rl.rl_interface.RL_interface.extract_per_hit_context` |
| `extract_per_hit_context` | `self`, `hit_data: dict` | `np.ndarray` | `rl_pretrain.RLPretrainer.run`<br>`rl_pretrain_premium.PremiumRLPretrainer.run` |

---

## `combine_all_new/rl_pretrain.py`

### Classes

#### `RLPretrainer`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `pretrainEpochs: int = 1`, `scoreThreshold: float = 15.0`, `convertedPtDir: str = CONVERTED_PT_DIR`, `outputBanditPath: str = OUTPUT_BANDIT_PATH`, `baseModelPath: str \| None = None`, `pretrainedBanditPath: str \| None = None`, `dryRun: bool = False` | `None` | `rl_pretrain.py::<module>` |
| `_computeBanditMseLoss` | `self`, `buffers: list` | `float` | `rl_pretrain.RLPretrainer.run` |
| `_savePlot` | `self`, `outputDir: str` | `None` | `rl_pretrain.RLPretrainer.run` |
| `run` | `self` | `str` | `rl_pretrain.py::<module>` |

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `setSeed` | `seed: int = SEED` | `None` | `rl_pretrain.py::<module>` |

---

## `combine_all_new/rl_pretrain_premium.py`

### Classes

#### `PremiumRLPretrainer`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `pretrainEpochs: int = DEFAULT_PRETRAIN_EPOCHS`, `convertedPtDir: str = CONVERTED_PT_DIR`, `outputBanditPath: str = OUTPUT_BANDIT_PATH` | `None` | `rl_pretrain_premium.py::<module>` |
| `run` | `self` | `None` | `rl_pretrain_premium.py::<module>` |
| `_evaluate_mse` | `self`, `datasets_by_action: dict` | `float` | `rl_pretrain_premium.PremiumRLPretrainer.run` |
| `_plot_convergence` | `self`, `train_history: list`, `valid_history: list` | `None` | `rl_pretrain_premium.PremiumRLPretrainer.run` |

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `setSeed` | `seed: int = SEED` | `None` | `rl_pretrain_premium.py::<module>` |

---

## `combine_all_new/test_rl_all.py`

### Classes

#### `TestBanditCore(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `setUp` | `self` | `None` | — |
| `test_select_action_valid` | `self` | `None` | — |
| `test_update_fills_buffer` | `self` | `None` | — |
| `test_compute_reward_clipped` | `self` | `None` | — |
| `test_save_load_roundtrip` | `self` | `None` | — |
| `test_pretrained_policy_clean_signal` | `self` | `None` | — |
| `test_pretrained_policy_noisy_signal` | `self` | `None` | — |

#### `TestUpdateScheduler(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `test_trigger_on_threshold` | `self` | `None` | — |
| `test_reset_clears_score` | `self` | `None` | — |
| `test_action_scores_ordering` | `self` | `None` | — |

#### `TestContextNormalizer(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `test_output_in_unit_range` | `self` | `None` | — |
| `test_repeated_same_input_converges` | `self` | `None` | — |

#### `TestRLInterface(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `setUp` | `self` | `None` | — |
| `test_extract_context_shape` | `self` | `None` | — |
| `test_entropy_range` | `self` | `None` | — |
| `test_full_pipeline_context_to_action` | `self` | `None` | — |

#### `TestRLPretrainSimulator(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `test_reward_ordering_clean_signal` | `self` | `None` | — |
| `test_reward_ordering_noisy_signal` | `self` | `None` | — |
| `test_pretrained_model_exists_and_loadable` | `self` | `None` | — |

#### `TestRLLoopWorker(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `setUp` | `self` | `None` | — |
| `test_snapshot_shape` | `self` | `None` | — |
| `test_context_and_action` | `self` | `None` | — |
| `test_update_scheduler_accumulates` | `self` | `None` | — |
| `test_phase2_reward_and_update` | `self` | `None` | — |
| `test_bandit_save_after_update` | `self` | `None` | — |

#### `TestIntermissionCalib(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `setUp` | `self` | `None` | — |
| `test_delta_sync` | `self` | `None` | — |
| `test_all_records_updated` | `self` | `None` | — |
| `test_reward_direction_on_improvement` | `self` | `None` | — |
| `test_bandit_save_after_calib` | `self` | `None` | — |

#### `TestExperienceReplayMemory(unittest.TestCase)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `setUp` | `self` | `None` | — |
| `test_add_and_sample_dynamic` | `self` | `None` | — |
| `test_dynamic_size_limit` | `self` | `None` | — |
| `test_save_load_buffer` | `self` | `None` | — |

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `make_rl` | — | `RL_interface` | `test_rl_all.TestIntermissionCalib.setUp`<br>`test_rl_all.TestRLInterface.setUp`<br>`test_rl_all.TestRLLoopWorker.setUp` |
| `make_hit_data` | — | `dict` | `test_rl_all.TestRLInterface.test_entropy_range`<br>`test_rl_all.TestRLInterface.test_extract_context_shape`<br>`test_rl_all.TestRLInterface.test_full_pipeline_context_to_action` |
| `tearDownModule` | — | `None` | — |

---

## `combine_all_new/utils/log_class.py`

### Classes

#### `Tee`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `*files: IO` | `None` | `core.app_controller.MainController.before_start`<br>`rl_pretrain.py::<module>`<br>`rl_pretrain_premium.py::<module>`<br>`utils.log_class.py::<module>` |
| `write` | `self`, `obj: str` | `None` | — |
| `flush` | `self` | `None` | — |

---
