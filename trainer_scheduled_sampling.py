from collections import defaultdict, deque
import random

    def __init__(
        self,
        model: nn.Module,
        trainer_config: TrainerConfig,
        log_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # ===================== Scheduled Sampling =====================
        use_scheduled_sampling: bool = True,
        teacher_forcing_start: float = 1.0,
        teacher_forcing_end: float = 0.05,
        schedule_type: str = "inverse_sigmoid",
        schedule_k: float = 0.015,
        num_teacher_steps: int = 2,           # сколько последних лагов заменяем (рекомендую 1 или 2)
        target_feature_idx: int = -1,         # индекс target в размерности F
    ):
        # ... твой существующий код до self.loss_fn ...

        self.use_scheduled_sampling = use_scheduled_sampling
        self.teacher_forcing_start = teacher_forcing_start
        self.teacher_forcing_end = teacher_forcing_end
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k
        self.num_teacher_steps = num_teacher_steps
        self.target_feature_idx = target_feature_idx

        self.scenario_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.num_teacher_steps)
        )

    def reset_scenario_buffer(self):
        """Сбрасываем буфер перед каждой эпохой"""
        self.scenario_buffer.clear()

    def _get_teacher_prob(self, epoch: int, total_epochs: int) -> float:
        if not self.use_scheduled_sampling:
            return 1.0

        if self.schedule_type == "inverse_sigmoid":
            return self.schedule_k / (self.schedule_k + np.exp((epoch - total_epochs / 2) / (total_epochs / 10)))
        elif self.schedule_type == "linear":
            progress = epoch / max(total_epochs - 1, 1)
            return max(self.teacher_forcing_end,
                      self.teacher_forcing_start - progress * (self.teacher_forcing_start - self.teacher_forcing_end))
        return 0.5

    def _process_single_scenario_with_buffer(
        self,
        scenario_name: str,
        x_seq: torch.Tensor,      # (1, 4, X, Y, Z, F)
        y_target: torch.Tensor,   # (1, X, Y, Z, C)
        mask: Optional[torch.Tensor],
        teacher_prob: float,
        is_training: bool = True,
    ):
        if random.random() < teacher_prob or not is_training:
            # Teacher Forcing
            outputs = self.model(x_seq, mask=mask)
        else:
            # Scheduled Sampling — заменяем последние N лагов из буфера
            x_modified = x_seq.clone()
            buffer = self.scenario_buffer[scenario_name]

            if len(buffer) > 0:
                # Заменяем последние min(N, len(buffer)) лагов
                for i, pred_value in enumerate(buffer, start=1):
                    if i > self.num_teacher_steps:
                        break
                    x_modified[0, -i, ..., self.target_feature_idx] = pred_value

            outputs = self.model(x_modified, mask=mask)

        pred = outputs["pred"]  # (1, X, Y, Z, C)

        # === Обновляем буфер новым предсказанием ===
        target_pred = pred.detach().squeeze(-1) if pred.shape[-1] == 1 else pred[..., 0].detach()
        self.scenario_buffer[scenario_name].appendleft(target_pred)  # новое предсказание в начало

        return outputs

    def process_batch_scheduled(
        self,
        batch: Any,
        epoch: int,
        total_epochs: int,
        is_training: bool = True,
    ) -> Tuple[float, Optional[Dict], float]:
        
        x_seq, y_target, mask, _, scenario_names = batch
        x_seq = x_seq.to(self.device)
        y_target = y_target.to(self.device)
        mask = mask.to(self.device) if mask is not None else None

        teacher_prob = self._get_teacher_prob(epoch, total_epochs)
        total_loss = 0.0
        outputs_list = []

        for i, scenario_name in enumerate(scenario_names):
            x_i = x_seq[i:i+1]
            y_i = y_target[i:i+1]
            mask_i = mask

            outputs = self._process_single_scenario_with_buffer(
                scenario_name=scenario_name,
                x_seq=x_i,
                y_target=y_i,
                mask=mask_i,
                teacher_prob=teacher_prob,
                is_training=is_training,
            )
            pred = outputs["pred"]

            per_elem_loss = self.loss_fn(pred, y_i)
            if mask is not None:
                mask_exp = mask.unsqueeze(0).unsqueeze(-1).expand_as(per_elem_loss)
                loss_i = (per_elem_loss * mask_exp).sum() / mask_exp.sum().clamp(min=1.0)
            else:
                loss_i = per_elem_loss.mean()

            total_loss += loss_i.item()
            outputs_list.append(outputs)

        avg_loss = total_loss / len(scenario_names)

        if is_training:
            self.optimizer.zero_grad()
            loss_tensor = torch.tensor(avg_loss, device=self.device, requires_grad=True)
            self.scaler.scale(loss_tensor).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if (self.scheduler is not None and 
                self.trainer_config.scheduler_config.type == SchedulerType.ONE_CYCLE):
                self.scheduler.step()

            return avg_loss, None, teacher_prob
        else:
            combined = {"pred": torch.cat([o["pred"] for o in outputs_list], dim=0)}
            return avg_loss, combined, teacher_prob

  def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int, total_epochs: int) -> float:
        self.model.train()
        self.reset_scenario_buffer()                    # важно!
        
        total_loss = 0.0
        total_tf = 0.0
        n = 0

        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in pbar:
            n += 1
            loss, _, tf_prob = self.process_batch_scheduled(
                batch=batch, epoch=epoch, total_epochs=total_epochs, is_training=True
            )
            total_loss += loss
            total_tf += tf_prob
            pbar.set_postfix({"loss": f"{total_loss/n:.6f}", "tf_prob": f"{total_tf/n:.3f}"})

        return total_loss / n
