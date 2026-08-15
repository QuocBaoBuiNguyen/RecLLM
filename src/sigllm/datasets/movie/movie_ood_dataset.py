"""Placeholder concrete dataset for MovieDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np

from sigllm.common.config import Config
from sigllm.common.logging_utils import NotebookLogger
from sigllm.datasets.base.rec_base_dataset import RecBaseDataset

LOGGER = NotebookLogger.rich_logger("sigllm.movie_ood_dataset")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

class MovieOODDataset(RecBaseDataset):

	# Length of the history window the loader actually feeds the model (the last
	# MAX_HISTORY_LEN ids; CoLLM uses the same cap in rec_datasets.py).
	MAX_HISTORY_LEN = 10

	def __init__(
		self,
		dataset_config,
		filename: str = None,
		subset: Literal["all", "warm", "cold"] = "all",
		train_item_ids: Optional[set] = None,
	) -> None:
		ann_path = Path(dataset_config.build_info.storage) / filename
		
		if (ann_path is None) or (not ann_path.exists()):
			raise ValueError(f"Annotation path {ann_path} does not exist.")
		
		df = pd.read_pickle(ann_path.with_suffix(".pkl")).reset_index(drop=True)
		self.annotation = df.copy()

		# SeLLa-matched warm/cold partition (BUG 2, Mismatch 1). SeLLa
		# (prepare_finetune_data.py:142) partitions the WHOLE test set by a single
		# `not_cold` flag: warm = not_cold==1, cold = not_cold==0. The old code
		# used a separate, stricter `warm` column (user AND item each with >3 train
		# interactions), a smaller/different population that is NOT comparable to
		# SeLLa's Table 3 warm. Use not_cold for both so the split matches SeLLa.
		if subset == "warm":
			self.annotation = df[df['not_cold'].isin([1])].copy()

		if subset == "cold":
			self.annotation = df[df['not_cold'].isin([0])].copy()

		# no-history-drop policy — TOGGLEABLE to match either reference:
		#   * SeLLa (codes/step3_train_sella/prepare_finetune_data.py:47-51) DROPS
		#     rows whose his_title has <2 entries, on EVERY split. Easier population.
		#   * CoLLM (minigpt4/datasets/datasets/rec_datasets.py:49,82-83) does NOT
		#     drop anything — it KEEPS all rows and zero-pads short histories.
		# The two references therefore evaluate on DIFFERENT test populations, so
		# our numbers are only comparable to whichever we mirror. Gate on
		# build_info.match_sella_history_filter (default True = SeLLa; set False in
		# the config to reproduce CoLLM's Table numbers on the full population).
		match_sella_history_filter = bool(
			getattr(dataset_config.build_info, "get", lambda *a: True)(
				"match_sella_history_filter", True
			)
		)
		if match_sella_history_filter and 'his_title' in self.annotation.columns:
			_before = len(self.annotation)
			self.annotation = self.annotation[
				self.annotation['his_title'].map(lambda t: len(t) >= 2)
			].reset_index(drop=True)
			log_step(
				"SeLLa his_title>=2 filter",
				f"subset={subset}: {_before} -> {len(self.annotation)} rows",
			)
		elif 'his_title' in self.annotation.columns:
			log_step(
				"CoLLM-parity: his_title>=2 filter DISABLED",
				f"subset={subset}: keeping all {len(self.annotation)} rows (zero-pad short history)",
			)

		self.use_his = False
		self.prompt_flag = False
		# Carried through only so the P1 content bridge can rebuild Stage-1's
		# "Title: X. Genres: Y." string. Unused when the model flag is off; an
		# extra string field in the sample changes nothing numerically.
		self.has_genres = 'genres' in self.annotation.columns

		if "sessionItems" in self.annotation.columns or "his" in self.annotation.columns:
			used_columns = ['uid','iid','title','his', 'his_title','label']
			renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']

			if self.has_genres:
				used_columns.append('genres')
				renamed_columns.append('TargetItemGenres')

			if 'not_cold' in self.annotation.columns:
				used_columns.append('not_cold')
				renamed_columns.append('prompt_flag')
				self.prompt_flag = True
			
			self.use_his = True
			self.annotation = self.annotation[used_columns]
			self.annotation.columns = renamed_columns
			
			self.annotation['InteractedItemIDs'] = self.annotation['InteractedItemIDs'].map(list)
			self.annotation['InteractedItemTitles'] = self.annotation['InteractedItemTitles'].map(list)
		else:
			used_columns = ['uid','iid','title','label']
			renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']

			if self.has_genres:
				used_columns.append('genres')
				renamed_columns.append('TargetItemGenres')
			if 'not_cold' in self.annotation.columns:
				used_columns.append('not_cold')
				renamed_columns.append('prompt_flag')
				self.prompt_flag = True
			
			self.annotation = self.annotation[used_columns]
			self.annotation.columns = renamed_columns
		
		# P0 cold-ITEM gating. Opt-in: the builder passes `train_item_ids` only
		# when `build_info.mark_cold_items` is set, so the vanilla path is
		# untouched when the flag is off.
		#
		# WHY this exists: the MF teacher is trained on train_ood2 ONLY, so an
		# item absent from train never receives a real gradient. Adam's coupled
		# weight_decay still decays those rows every step (nn.Embedding produces
		# DENSE grads), so they end up ~0 — measured on the ML-1M MF ckpt,
		# trained item rows have norm mean 0.749 vs 0.038 for untrained ones,
		# and untrained USER rows are exactly 0.0. proj_cf(~0) is then the same
		# constant token for EVERY cold item, so the CF channel cannot separate
		# two cold items of the same user. That is what flattens cold uAUC
		# (book: test_cold 0.5258 vs test_warm 0.6238) while global AUC survives
		# on cross-user spread.
		#
		# NB: deliberately keyed on the ITEM, not on `not_cold`/`prompt_flag`.
		# `not_cold` = (uid in train) AND (iid in train); on Amazon-Book that is
		# ~item-cold (7,545 item-cold of 8,308 not_cold==0 test rows), but on
		# ML-1M it is dominated by cold-USER (42.4% of test rows vs only 1.3%
		# item-cold), so reusing it would mislabel 42% of the movie rows.
		self.mark_cold_items = train_item_ids is not None
		if self.mark_cold_items:
			pad = 0
			self.annotation['TargetItemIsCold'] = (
				~self.annotation['TargetItemID'].isin(train_item_ids)
			).astype(int)

			cold_hist = 0
			total_hist = 0
			if self.use_his:
				# Remap cold history ids to the padding index so the existing
				# `ids != padding_index` mask in the model drops those soft
				# tokens outright (10.0% of book history tokens, 0.8% on ML-1M)
				# instead of injecting a constant. The TITLES are intentionally
				# left in place: the text is informative, only the untrained CF
				# vector is not.
				def _pad_cold_history(ids_):
					nonlocal cold_hist, total_hist
					# Remap the whole list, but COUNT only inside the last
					# MAX_HISTORY_LEN window: that is what __getitem__ keeps, so
					# a rate over the full history would understate the share of
					# soft tokens actually affected (ML-1M: 0.6% full vs 0.8%
					# in-window; book: 10.0% in-window).
					window_start = max(0, len(ids_) - self.MAX_HISTORY_LEN)
					out = []
					for pos, x in enumerate(ids_):
						if x == pad:
							out.append(pad)
							continue
						is_cold_item = x not in train_item_ids
						if pos >= window_start:
							total_hist += 1
							cold_hist += int(is_cold_item)
						out.append(pad if is_cold_item else x)
					return out

				self.annotation['InteractedItemIDs'] = (
					self.annotation['InteractedItemIDs'].map(_pad_cold_history)
				)

			_n_cold = int(self.annotation['TargetItemIsCold'].sum())
			_n_rows = len(self.annotation)
			_detail = (
				f"subset={subset}: target cold {_n_cold}/{_n_rows} rows "
				f"({100.0 * _n_cold / max(_n_rows, 1):.1f}%)"
			)
			if self.use_his:
				_detail += (
					f", history ids cold {cold_hist}/{max(total_hist, 1)} "
					f"({100.0 * cold_hist / max(total_hist, 1):.1f}%) -> padded"
				)
			log_step("Cold-item gating ACTIVE", _detail)

		log_step("data path", f"{ann_path} | data size: {self.annotation.shape}")
		self.user_num = self.annotation['UserID'].max() + 1
		self.item_num = self.annotation['TargetItemID'].max() + 1

		if self.use_his:
			max_length = 0
			for his in self.annotation['InteractedItemIDs']:
				max_length = max(max_length, len(his))
			self.max_length = min(max_length, self.MAX_HISTORY_LEN)
			log_step("Movie OOD datasets, max history length:", str(self.max_length))
	
	def __getitem__(self, index):
		
		row = self.annotation.iloc[index]

		def _add_aux_fields(sample: dict) -> dict:
			if self.prompt_flag:
				sample["prompt_flag"] = row["prompt_flag"]
			if self.mark_cold_items:
				# Consumed by QRecLLM when model.cold_item_token=True.
				sample["TargetItemIsCold"] = int(row["TargetItemIsCold"])
			if self.has_genres:
				# Consumed by QRecLLM when qformer_config.item_text_instruction=True.
				sample["TargetItemGenres"] = str(row["TargetItemGenres"]).replace("|", ", ")
			return sample

		user_id = row["UserID"]
		target_item_id = row["TargetItemID"]
		target_title = row["TargetItemTitle"].strip(" ")
		label = row["label"]
		
		if self.use_his:
			history_item_ids = row["InteractedItemIDs"]
			history_titles = row["InteractedItemTitles"]
			history_len = len(history_item_ids)

			interacted_count = history_len - 1 if (history_len > 0 and history_item_ids[0] == 0) else history_len

			max_history_len = self.max_length  

			if history_len < max_history_len:
				pad_size = max_history_len - history_len
				padded_history_ids = ([0] * pad_size) + list(history_item_ids)
			elif history_len > max_history_len:
				padded_history_ids = list(history_item_ids[-max_history_len:])
				interacted_count = max_history_len
			else:
				padded_history_ids = list(history_item_ids)

			recent_titles = history_titles[-interacted_count:] if interacted_count > 0 else []
			processed_titles = self.convert_title_list(recent_titles)

			sample = {
				"UserID": user_id,
				"InteractedItemIDs_pad": np.array(padded_history_ids),
				"InteractedItemTitles": processed_titles,
				"TargetItemID": target_item_id,
				"TargetItemTitle": f"\"{target_title}\"",
				"InteractedNum": interacted_count,
				"label": row["label"],
			}
			return _add_aux_fields(sample)
		else:
			sample = {
				"UserID": user_id,
				"TargetItemID": target_item_id,
				"TargetItemTitle": target_title,
				"label": label,
			}
			return _add_aux_fields(sample)
		
