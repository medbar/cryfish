import torch


def zip_embeddings(
    prompt_list_embs, audiollm_embeds, audio_id2sample_id, audiollm_attention_mask
):
    """
    Merges embeddings into a sequence:
    prompt_embs[0][0], audiollm_embeds[0], prompt_embs[0][1], audiollm_embeds[1], ..., prompt_embs[0][-1]
    Minimizes the amount of padding in the batch
    Args:
        prompt_list_embs (List[torch.Tensor]): [[left_prompt_embs, delta_embs, ..., right_prompt_embs], ...]
        audiollm_embeds (torch.Tensor): Audio embeddings [B+C, T, D]
        audio_id2sample_id (List[int]): Indices mapping to group embeddings by sample
        audiollm_attention_mask (torch.Tensor, optional): Attention mask for audio [B+C, T]

    Returns:
        tuple: (inputs_embeds, input_attention_mask)
    """
    device = audiollm_embeds.device
    B = len(prompt_list_embs)
    D = audiollm_embeds.shape[-1]

    # Group embeddings by sample_id, removing padding
    grouped_embeds = [[] for _ in range(B)]
    for i, sample_id in enumerate(audio_id2sample_id):
        grouped_embeds[sample_id].append(audiollm_embeds[i][audiollm_attention_mask[i]])

    assert all(
        len(g) + 1 == len(p) for g, p in zip(grouped_embeds, prompt_list_embs)
    ), f"{[(len(g), len(p)) for g, p in zip(grouped_embeds, prompt_list_embs)]}"

    lens = [
        sum(len(a) for a in audio_embeddings) + sum(len(p) for p in prompt_embeddings)
        for audio_embeddings, prompt_embeddings in zip(grouped_embeds, prompt_list_embs)
    ]
    T = max(lens)
    # Build sequences for each batch
    # Keep padding at the BEGINNING of the batch
    all_embeds = audiollm_embeds.new_zeros(B, T, D)
    for i, (audio_embeddings, prompt_embeddings) in enumerate(
        zip(grouped_embeds, prompt_list_embs)
    ):
        j = T - lens[i]
        next_j = j + len(prompt_embeddings[0])
        all_embeds[i, j:next_j] = prompt_embeddings[0]
        j = next_j
        for a, p in zip(audio_embeddings, prompt_embeddings[1:]):
            next_j = j + len(a)
            all_embeds[i, j:next_j] = a
            j = next_j
            next_j = j + len(p)
            all_embeds[i, j:next_j] = p
            j = next_j
    # Build attention masks for each batch
    all_attention_masks = torch.arange(T, device=device) >= (
        T - torch.as_tensor(lens, device=device).unsqueeze(1)
    )

    return all_embeds, all_attention_masks


def test_zip_embeddings():
    # Create test data
    D = 4  # размерность эмбеддингов

    # Create prompt embeddings for two samples
    prompt_list_embs = [
        [torch.ones(2, D), torch.ones(1, D) * 2, torch.ones(2, D) * 3],  # first sample
        [torch.ones(1, D), torch.ones(2, D) * 2],  # second sample
    ]

    # Create audio embeddings (3 audios, 2 time steps)
    audiollm_embeds = torch.ones(3, 2, D) * 4
    # Audio mask (True - real data, False - padding)
    audiollm_attention_mask = torch.tensor(
        [
            [True, False],  # first audio: only 1 time step
            [True, True],  # second audio: both time steps
            [False, True],  # third audio: only 1 time step
        ]
    )

    # Mapping audio to samples: first sample has 2 audios, second has 1 audio
    audio_id2sample_id = [0, 0, 1]

    # Call the function under test
    all_embeds, all_attention_masks = zip_embeddings(
        prompt_list_embs, audiollm_embeds, audio_id2sample_id, audiollm_attention_mask
    )

    # Check output tensor shapes
    assert all_embeds.shape[0] == 2  # batch size
    assert all_embeds.shape[2] == D  # embedding dimension
    assert all_attention_masks.shape == all_embeds.shape[:2]

    # Check correctness of masks
    assert all_attention_masks.sum(1).tolist() == [
        8,
        4,
    ]  # number of real tokens in each sample

    print("Passed!")
