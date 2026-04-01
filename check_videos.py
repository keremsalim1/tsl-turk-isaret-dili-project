import os

d = r"C:\Users\ASUS\Desktop\trainset\train_set_vfbha39\train"

files = os.listdir(d)
vids = [f for f in files if f.lower().endswith(('.mp4','.avi','.mov','.mkv','.webm'))]
dirs = [f for f in files if os.path.isdir(os.path.join(d, f))]

print("Toplam dosya:", len(files))
print("Video dosyasi:", len(vids))
print("Alt klasor:", len(dirs))

if dirs:
    print("\nIlk 10 klasor:")
    for x in sorted(dirs)[:10]:
        sub = os.listdir(os.path.join(d, x))
        print(f"  {x}/ ({len(sub)} dosya)")

print("\nIlk 30 video:")
for v in sorted(vids)[:30]:
    print(f"  {v}")
