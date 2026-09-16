import torch
import torch.nn.functional as F


def plackett_luce_log_probs(scores: torch.Tensor) -> torch.Tensor:
    """Return Plackett-Luce log-probabilities for the ranking induced by scores."""
    ranking = scores.argsort(dim=-1, descending=True)
    ordered_scores = scores.gather(1, ranking)
    log_probs = ordered_scores - torch.logcumsumexp(ordered_scores.flip(-1), dim=-1).flip(-1)
    result = torch.zeros_like(log_probs)
    return result.scatter(1, ranking, log_probs)


def vpd_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, beta: float = 1.0):
    """Vanilla Preference Distillation: NLL of the teacher-induced ranking."""
    ranking = teacher_scores.argsort(dim=-1, descending=True)
    student_scores = beta * student_scores
    ordered_student_scores = student_scores.gather(1, ranking)
    log_probs = ordered_student_scores - torch.logcumsumexp(
        ordered_student_scores.flip(-1), dim=-1
    ).flip(-1)
    return -log_probs.sum(dim=-1)


def ppd_loss(student_scores: torch.Tensor, teacher_scores: torch.Tensor, beta: float = 1.0):
    """Probabilistic Preference Distillation: JSD of Plackett-Luce distributions."""
    student_log_probs = plackett_luce_log_probs(beta * student_scores)
    teacher_log_probs = plackett_luce_log_probs(beta * teacher_scores)
    mixture_log_probs = torch.logaddexp(student_log_probs, teacher_log_probs) - torch.log(
        torch.tensor(2.0, device=student_scores.device, dtype=student_scores.dtype)
    )
    student_kl = F.kl_div(mixture_log_probs, student_log_probs.exp(), reduction="none").sum(-1)
    teacher_kl = F.kl_div(mixture_log_probs, teacher_log_probs.exp(), reduction="none").sum(-1)
    return 0.5 * (student_kl + teacher_kl)
