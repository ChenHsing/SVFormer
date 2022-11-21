import torch
import torch.nn as nn


class MemoryStore(object):

    def __init__(self, memory_size, n_classes, n_neighbours, embed_dim, device):
        self.memory_size = memory_size
        self.n_classes = n_classes
        self.n_neighbours = n_neighbours
        self.embed_dim = embed_dim
        self.device = device

        # Initializing memory to blank embeddings and "n_classes = not seen" labels
        self.memory_embeds = torch.zeros(
            (self.memory_size, self.embed_dim)).to(self.device)
        self.memory_labels = self.n_classes * \
            torch.ones((self.memory_size,), dtype=torch.long).to(self.device)
        self.write_pointer = 0
        self.added_memories = 0

    def __len__(self):
        return min(self.memory_size, self.added_memories)

    def add_entry(self, embed, label):
        """
        Args:
            embed: a torch tensor with (1, embed_dim) size
            label: a torch tensor with (1) size
        Returns:
            None
        """
        self.memory_embeds[self.write_pointer] = embed
        self.memory_labels[self.write_pointer] = label
        self.write_pointer = (self.write_pointer + 1) % self.memory_size
        self.added_memories += 1

    def add_batched_entries(self, embeds, labels):
        """
        Args:
            embed: a torch tensor with (batch, embed_dim) size
            label: a torch tensor with (batch) size
        Returns:
            None
        """
        if self.memory_size - self.write_pointer >= embeds.shape[0] or embeds.shape[0] == 1:
            start = self.write_pointer
            end = self.write_pointer + embeds.shape[0]
            self.memory_embeds[start:end] = embeds
            self.memory_labels[start:end] = labels
            self.write_pointer = (self.write_pointer +
                                  embeds.shape[0]) % self.memory_size
        else:
            remaining_space = self.memory_size - self.write_pointer
            overflow_space = embeds[remaining_space:].shape[0]
            start = self.write_pointer
            end = self.memory_size
            self.memory_embeds[start:end] = embeds[:remaining_space]
            self.memory_embeds[0:overflow_space] = embeds[remaining_space:]
            self.memory_labels[start:end] = labels[:remaining_space]
            self.memory_labels[0:overflow_space] = labels[remaining_space:]
            self.write_pointer = overflow_space
        self.added_memories += embeds.shape[0]

    def get_nearest_entries(self, queries):
        """
        Reference code for pairwise distance computation:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
        Args:
            queries: a torch array with (n_queries, embed_dim) size
        Returns:
            embeds: a torch array with (n_queries, max_neighbours, embed_dim) size
            labels: a torch array with (n_queries, max_neighbours) size
        """
        # Masking not filled memories
        mask_idx = min(self.memory_size, max(
            self.n_neighbours, self.added_memories))
        mask_memory = self.memory_embeds[:mask_idx]

        # Calculating pairwise distances between queries (q) and memory (m) entries
        q_norm = torch.sum(queries ** 2, dim=1).view(-1, 1)
        m_norm = torch.sum(mask_memory ** 2, dim=1).view(1, -1)
        qm = torch.mm(queries, mask_memory.transpose(1, 0))
        dist = q_norm - 2 * qm + m_norm

        # Determining indices of nearest memories and fetching corresponding labels and embeddings
        distances, idx = torch.topk(-dist, dim=1, k=self.n_neighbours)
        distances = -1.0 * distances
        return self.memory_embeds[idx], self.memory_labels[idx], distances

    def flush(self):
        self.memory_embeds = torch.zeros(
            (self.memory_size, self.embed_dim)).to(self.device)
        self.memory_labels = self.n_classes * \
            torch.ones((self.memory_size,), dtype=torch.long).to(self.device)
        self.write_pointer = 0
        self.added_memories = 0