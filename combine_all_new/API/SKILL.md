# SKILL: Python Codebase API + Call-Graph Documentation Generator

## 目標

掃描專案中所有 `.py` 檔案，輸出單一 `API_MAP.md`，每個 function / method 同時揭露：

1. **完整 signature**：所有 parameter 的型別 + 預設值、return type。
2. **直接呼叫者（Called By）**：這個 function / method 在專案內被誰呼叫。

`API_MAP.md` 是「打開來看哪個 method 長怎樣 + 誰會打它」的單一參考檔。**不再分檔**（先前的 `FUNC_CALL.md` 已合併進來）。

---

## 執行步驟（兩階段：先掃 signature，再掃 call graph，最後合併輸出）

### Step 1 — 收集所有 `.py` 檔案

從專案根目錄遞迴搜尋所有 `.py` 檔案，排除：`__pycache__`、`.git`、`venv`、`env`、`.venv`、`node_modules`、`.tox`、`wait_to_use`、`env_setup`、`Historical_design_process`。結果依照目錄結構排序。

### Step 2 — 解析每個檔案的 signatures（AST）

對每個 `.py` 檔案用 `ast.parse`：

1. **辨識 class**：記錄 class 名稱與繼承的 parent class（如 `class Foo(Bar, Baz)`）。Enum 類別（繼承自 `Enum`/`IntEnum`/`Flag`/`IntFlag`/`StrEnum`）不展開 method 表格，列舉成員即可。
2. **辨識 function / method**（含 `def` 與 `async def`）：記錄名稱、所屬 class（若為 method）。
3. **解析簽名**：
   - 所有 parameters：名稱、type annotation（用 `ast.unparse`）、預設值、是否 `*args` / `**kwargs`。
   - Return type annotation。

### Step 3 — 推論缺漏的型別

若 parameter 缺 type annotation，依以下優先序推論：

#### 3.1 內部 class / 慣用名稱對照表（**最高優先**）

| Parameter 名稱     | 推論型別                  |
|--------------------|---------------------------|
| `osu_interface`    | `osu_interface`           |
| `RL_interface`     | `RL_interface`            |
| `model_interface`  | `model_interface`         |
| `tcp_client`       | `TCPClient \| None`       |
| `tcp_server`, `server` | `TCPServer`           |
| `bandit`           | `NeuralContextualBandit \| None` |
| `inlet`            | `StreamInlet \| None`     |
| `on_message`       | `Callable \| None`        |
| `rl_loop_worker_fn`| `Callable`                |

#### 3.2 通用命名慣例

| 名稱模式 | 推論型別 |
|----------|----------|
| `path`、`*_path`、`filepath`、`save_path`、`csv_path`、`checkpoint_path`、`base_path`、`map_file` | `str` |
| `name`、`*_name`、`account_name`、`model_name`、`text`、`msg`、`message`、`obj` | `str` |
| `idx`、`*_idx`、`index`、`mode`、`size`、`new_size`、`score`、`hits`、`key`、`lane`、`opt`、`version`、`count`、`note_count`、`max_width`、`current_note_index`、`direction`、`x`、`y`、`w`、`h`、`seed`、`fs`、`new_fs`、`n_chans`、`n_outputs`、`n_times`、`batch_size`、`num_epochs`、`epochs`、`patience`、`pretrainEpochs`、`update_times`、`current_offset` | `int` |
| `time_ms`、`game_time_ms`、`avg_diff`、`beat_progress`、`speed`、`accuracy`、`reward`、`low`、`high`、`lr`、`valid_ratio` | `float` |
| `fullscreen`、`active`、`filled`、`freeze_layer`、`ft`、`dryRun`、`RL_open`、`RL_LOOP_USE` | `bool` |
| `data`、`*_data`、`eeg_data`、`freqs`、`psd`、`sample_arr`、`epoch`、`segment`、`replay_segments`、`replay_labels` | `np.ndarray` |
| `probability` | `np.ndarray \| None` |
| `loader` | `DataLoader` |
| `dataset`、`*_dataset`、`val_dataset`、`dataset_val`、`dataset_valid` | `TensorDataset` |
| `model_class` | `type` |
| `notes`、`map_notes`、`history`、`account_list`、`model_list`、`song_list`、`*_list`、`train_history`、`valid_history`、`session_records`、`row` | `list` / `list[dict]` 視內容 |
| `rows` | `list[list]` |
| `settings`、`model_info`、`model_kwargs`、`song_meta`、`each_count`、`each_count_acc`、`hit_data`、`datasets_by_action`、`DATA_np_dict` | `dict` |
| `color` | `tuple[int, int, int]` |
| `surface` | `pygame.Surface` |
| `conn` | `socket.socket` |
| `addr` | `tuple` |
| `event` | `pygame.event.Event` |
| 尾綴 `_path`/`_name`/`_dir`/`_file`/`_id` | `str` |
| 尾綴 `_idx`/`_index`/`_count` | `int` |
| 尾綴 `_data` | `np.ndarray` |
| 尾綴 `_list` | `list` |
| 尾綴 `_dataset` | `TensorDataset` |
| 尾綴 `_kwargs` | `dict` |

#### 3.3 從 default value 反推

若上述都不命中，再看 default value：

- `ast.Constant` 為 `bool` → `bool`；`int` → `int`；`float` → `float`；`str` → `str`
- `ast.UnaryOp(USub/UAdd, Constant)`（例：`= -1`）→ 遞迴看 operand
- `ast.List` → `list`；`ast.Dict` → `dict`；`ast.Tuple` → `tuple`；`ast.Set` → `set`
- `None` → 留空（不夠資訊推 `X | None`）

#### 3.4 fallback

實在無法推斷 → 標記為 `Unknown`。`*args` / `**kwargs` 原樣保留（不加 type），唯一例外：`*files` → `*files: IO`。

#### 3.5 Return type

優先順序：

1. Source code 的 return annotation。
2. `__init__` → 一律 `None`。
3. `forward` → 一律 `torch.Tensor`（nn.Module 慣例）。
4. 沒有任何 `return` statement，或所有 `return` 都沒有值 → `None`。
5. 掃描函式內所有 `Return` node 的回傳值，逐一推論型別：
   - **常數 / 字面值**：`Constant` → 對應型別；`List` / `Dict` / `Tuple` / `Set` / `JoinedStr` → 對應型別。
   - **Call**：
     - 內建函式（`dict`、`list`、`len`、`sorted`、`isinstance` ...）
     - 已知 library（`np.*`、`torch.*`、`F.*`、`scipy.signal.*`、`pygame.*`、`glob.glob`、`json.load` ...）
     - 專案內 class 實例化 `ClassName(...)` → `ClassName`
     - `self.method()` → 用同檔案內已推論的該 method 回傳型別（pre-pass 先跑一遍）
   - **Attribute**：
     - `self.x` → 用 class scope 收集到的 `self.* = <expr>` 對照表
     - `KnownClass.MEMBER` → `KnownClass`（Enum）
   - **Name**（local 變數）：用 function 內 sequential pass 收集的 local var type 表。
   - **Subscript**：`x.shape[i]` / `x.size[i]` → `int`；`np.ndarray[...]` → `np.ndarray`。
   - **Compare / BoolOp(`and`/`or`)** → `bool`；**BinOp** → 依 operand 推（字串/數字/np）。
6. 對每個 `Return`，若推不出來則該 branch 標 Unknown；最後 union 所有 branch：
   - 全部相同 → 該型別
   - 有 `None` 配一個其他型別 → `T | None`
   - 一部分推得出 + 一部分不行 → `T | Unknown`
   - 全部推不出 → `Unknown`

#### 3.6 Local 變數 / `self.*` 屬性追蹤

為了讓 `return some_local` 與 `return self.attr` 能推論，要先掃描：

- **函式 body sequential pass**：`var = expr`、`var: T = expr`、`(a, b) = (X(), Y())`，把 `var → 型別` 寫進 local scope；參數本身的型別也先 seed 進 local scope。
- **Class 全體 pass**：把所有 method 內 `self.x = <expr>` 蒐集進 `self_types` 對照表（包含 class 實例化、常數、container literal、`Name` 指到 import 進來的 class、或對 `__init__` 參數的直接 forward）。後續任何 method 想推 `self.x` 的型別時就查這個表。

### Step 4 — 抽取呼叫關係（Call Graph，AST）

對每個 `.py` 檔案，掃描每個 function / method body 的所有 `Call` node，解析 callee 為「fully-qualified 名稱」：

- `Name("foo")` → 看 local scope → imports → module's own defs
- `Attribute(Name("X"), "y")`：若 `X` 是 imported module / class / 已知 local 變數，組對應符號
- `Attribute(Name("self"), "y")` → `CurrentClass.y`
- `Attribute(Attribute(...), "y")`：跨層 attribute（特別處理 `self.attr.method()`，需把整個 class 內 `self.* = SomeClass(...)` 的 assignment **預先掃過一遍**作為 method 共享的 `self.*` 型別資訊，否則例如 `self.level_runner.run_level(...)` 解不出來）

規則：

1. **只記錄專案內部呼叫**（symbol 在掃描範圍內定義），外部 library（torch / numpy / pygame / threading...）忽略。
2. **`ClassName(...)` 的實例化**視為呼叫 `ClassName.__init__`。
3. **動態呼叫**（callback、`threading.Thread(target=...)`、`getattr`、把 method 當參數傳）AST 抓不到，輸出 `—`，必要時人工補。

**Import 解析特例**：本專案 `main.py` 用 `sys.path.insert(0, combine_all_new)`，因此檔案內 `from core.X import Y` 實際指向 `combine_all_new.core.X.Y`。解析時若 import 路徑加上 `combine_all_new.` 前綴後對得上已知 module，視為內部 import。

把結果反向組成 `callee → [direct_callers]` 對照表。

### Step 5 — 合併輸出 `API_MAP.md`

將 signatures + call graph 合併為單一 markdown 表格，輸出至與此 SKILL.md 同層的 `API_MAP.md`。

#### 輸出格式範本

```markdown
# API Map

> 自動產生，掃描日期：YYYY-MM-DD
> 掃描範圍：`combine_all_new/`
>
> **欄位說明**
>
> - **Parameters**：`name: Type = default`；`self` / `cls` 省略型別。
> - **Return Type**：function 註解的回傳型別；`__init__` 視為 `None`，缺註解標 `Unknown`。
> - **Called By**：直接呼叫此 function 的位置（只追蹤 `combine_all_new/` 內部）。`ClassName()` 實例化視為呼叫 `__init__`；`::<module>` 表示 module 層級程式碼（含 `if __name__ == "__main__":`）。
>
> 缺型別註解時，會依參數名稱推論（包含專案內部 class 如 `osu_interface` / `model_interface` / `RL_interface` / `TCPServer` 等），無法推論才標 `Unknown`。

---

## `path/to/file.py`

### Classes

#### `ClassName(ParentA, ParentB)`

| Method | Parameters | Return Type | Called By |
|--------|-----------|-------------|-----------|
| `__init__` | `self`, `x: int`, `y: float = 0.0` | `None` | `path.to.other.module.fn` |
| `forward` | `self`, `data: np.ndarray` | `torch.Tensor` | `path.to.module.SomeClass.run` |

### Functions (module-level)

| Function | Parameters | Return Type | Called By |
|----------|-----------|-------------|-----------|
| `load_config` | `path: str`, `verbose: bool = False` | `dict` | `path.to.module.main` |
| `main` | — | `None` | `path.to.module.py::<module>` |

---
```

#### 格式規則

- 每個 `.py` 檔案用 `` ## `相對路徑` `` 作為二級標題。
- Class 用 `` #### `ClassName(Parents)` `` 標示（無 parent 時直接寫 `` `ClassName` ``）。
- **Enum class** 不展開 method 表格，列舉成員：`*(Enum 成員：A=0, B=1, C=2)*`。
- 每個 class 下方用表格列出所有 method（4 欄）。
- Module-level function 用獨立表格列出（4 欄）。
- Parameter 欄位：每個 param 用 backtick 包起來成 `` `name: Type = default` ``，與 Return Type / Called By 視覺一致；省略 `self` / `cls` 的型別但仍包 backtick。多個 param 用 `, ` 連接。
- 若無 parameter（只有 `self`），Parameters 欄位寫 `` `self` ``；若連 `self` 都沒有（純 module function），寫 `—`。
- Union 型別中的 `|` 要在 markdown table cell 內 escape 為 `\|`，否則會被當成欄位分隔。
- **Called By 欄位**：
  - 多個 caller 用 `<br>` 分隔，依字母序排列。
  - Caller 顯示去掉 `combine_all_new.` 前綴的相對路徑，例：`core.app_controller.MainController.start`。
  - Module 層級的呼叫（含 `if __name__ == "__main__":`）標為 `<file>.py::<module>`。
  - 沒被任何人呼叫的標 `—`（entry point、callback target、或 dead code，需人工判斷）。
- 空檔案或無 class / function 的檔案標 `*(無 class / function)*`。
- 遇 syntax error 的檔案：標記 `⚠️ Syntax Error` 並跳過。

---

## 注意事項

- **不要修改任何原始碼**，這是純讀取 + 文件產出的 pipeline。
- 每次執行此 SKILL 時，覆寫既有的 `API_MAP.md`。
- 若新增了 parameter naming convention（例如新的 interface class），請同步更新 Step 3.1 / 3.2 的對照表，再重新跑一次。
- 動態呼叫（callback、`threading.Thread(target=...)`、`getattr`）AST 抓不到 → Called By 欄留 `—`，這是預期行為而非 bug。
