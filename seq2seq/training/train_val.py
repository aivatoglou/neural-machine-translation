import torch


def train(model, iterator, optimizer, criterion, clip, device):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src, trg = (batch[0].to(device), batch[1].to(device))
        src, trg = torch.transpose(src, 0, 1), torch.transpose(trg, 0, 1)

        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src, trg = (batch[0].to(device), batch[1].to(device))

            src, trg = torch.transpose(src, 0, 1), torch.transpose(trg, 0, 1)

            output = model(src, trg)

            # Evaluate translation on each epoch
            to_return = output

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), to_return.argmax(2).transpose(0, 1)[-1]
