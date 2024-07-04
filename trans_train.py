from trans_main import *

torch.cuda.empty_cache()
step = 0
torch.backends.cudnn.benchmark = True
if model_checkpoint:
    step = int('model_10000.pt'.replace("model_", "").replace(".pt", ""))

model.train()
for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for index, data in enumerate(train_loader):
        # 生成数据
        src, tgt, tgt_y, n_tokens = data
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

        # 清空梯度
        optimizer.zero_grad()
        # 进行transformer的计算
        out = model(src, tgt)
        # 将结果送给最后的线性层进行预测
        out = model.predictor(out)

        """
        计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                (batch_size*词数, 词典大小)。
                而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                除以n_tokens。
        """
        loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        loop.set_description("Epoch {}/{}".format(epoch, epochs))
        loop.set_postfix(loss=loss.item())
        loop.update(1)

        step += 1

        del src
        del tgt
        del tgt_y

        if step != 0 and step % save_after_step == 0:
            torch.save(model, model_dir / f"model_{step}.pt")
