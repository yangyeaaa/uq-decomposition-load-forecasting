from common import *
d=prep("GEF"); model=make_model(d)
dt=train(model,d,seed=1,save=f"{OUT}/tmp.pth",epochs=3,patience=12)
print(f"3 epochs: {dt:.1f}s -> ~{dt/3:.1f}s/epoch; full 50-epoch run <= {dt/3*50/60:.1f} min")
print("trainable params (encoder+head):", sum(p.numel() for n,p in model.named_parameters() if not n.startswith("diffusion_net")))
r=predict(model,d["te"]); print(fmt(full_metrics(d["tel"],r["mu"],r["tot"],d["sdl"])))
