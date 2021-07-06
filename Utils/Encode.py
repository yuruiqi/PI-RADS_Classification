import torch


def encode(label):
    """
    label: (batch, 1) 2-5
    """
    label_code = label.repeat(1, 3)  # (batch, 3).
    zero = torch.tensor(0, device=label.device, dtype=label.dtype)
    one = torch.tensor(1, device=label.device, dtype=label.dtype)
    label_code[:, 0] = torch.where(label_code[:, 0]>2, one, zero)  # 3 to 100
    label_code[:, 1] = torch.where(label_code[:, 1]>3, one, zero)
    label_code[:, 2] = torch.where(label_code[:, 2]>4, one, zero)
    return label_code


def decode(prediction):
    """
    prediction: (batch, 3)
    output:
        prediction_r: (batch, 3), 0-3
    """
    zero = torch.tensor(0, device=prediction.device, dtype=prediction.dtype)
    one = torch.tensor(1, device=prediction.device, dtype=prediction.dtype)

    prediction = torch.where(prediction>0.5, one, zero)  # (batch, 3)
    prediction_r = torch.zeros([prediction.shape[0], 1])
    for i in range(3):
        prediction_r[prediction[:, 0] > 0] = one  # (batch, 1)
        prediction_r[prediction[:, 1] > 0] = torch.tensor(2, device=prediction.device,
                                                          dtype=prediction.dtype)  # (batch, 1)
        prediction_r[prediction[:, 2] > 0] = torch.tensor(3, device=prediction.device,
                                                          dtype=prediction.dtype)  # (batch, 1)
    return prediction_r


if __name__ == '__main__':
    # label = torch.tensor([[2],[3],[4],[5]])
    # print(encode(label))

    prediction = torch.tensor([[0.1,0.2,0.3],
                               [0.9,0.7,0.3],
                               [0.4,0.9,0.3],
                               [0.9,0.9,0.9],
                               [0.1,0.2,0.3],
                               [0.1,0.2,0.7],])
    print(decode(prediction))
