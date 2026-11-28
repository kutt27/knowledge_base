**Linux: Reset a User Password from a Live ISO**

The procedure requires the target machine and a live Linux ISO. The ISO is booted, the root filesystem is mounted, and the password hash is replaced. The ordinary case consists of three stages.

No step is specific to a distribution. Device names in the examples are placeholders and are replaced with those reported on the target.

**Boot the live ISO**

The ISO is booted on the target machine and a terminal is opened.

**Identify the disk**

```
lsblk -f
```

The required information is the root partition and its filesystem type, `/dev/nvme0n1p3` with ext4 in this example. The EFI partition is excluded.

Where the output reports `crypto_LUKS`, a logical volume, or btrfs, the optional block below is applied before mounting. Otherwise that block is skipped.

**Mount the root**

```
sudo mkdir -p /mnt
sudo mount /dev/nvme0n1p3 /mnt
```

If the mount reports that the filesystem is already mounted, it is released first.

```
sudo umount /dev/nvme0n1p3
```

The mounted system is confirmed as the intended one.

```
ls /mnt/etc/shadow
```

**Optional: encrypted, LVM, or btrfs root**

This block is skipped entirely when the mount above succeeds.

For a LUKS encrypted disk, the container is opened and the mapper device is mounted in place of the partition.

```
sudo cryptsetup luksOpen /dev/nvme0n1p3 crypt
```

For LVM, the volume group is activated and the logical volume is mounted.

```
sudo vgchange -ay
ls /dev/mapper
sudo mount /dev/mapper/vg-root /mnt
```

For a btrfs root, the subvolume is specified at mount time.

```
sudo mount -o subvol=@ /dev/nvme0n1p3 /mnt
```

The layers compose. Unlocking precedes activation, and the device present at the end of that sequence is the one mounted.

**Generate a hash**

```
openssl passwd -6 'NewPass123'
```

The resulting `$6$` string is copied in full.

**Replace the hash**

The credential is stored in `/etc/shadow`. The `x` field in `/etc/passwd` is only a pointer, so an edit to `/etc/passwd` has no effect.

```
sudo nano /mnt/etc/shadow
```

The line belonging to the target account is located by its first field. The text between the first and second colon is replaced with the generated hash. No other character on the line is altered.

A field beginning with `!` or `*` denotes a locked account. The entire field is replaced with the hash, and neither marker is retained.

**Correct the file attributes**

```
sudo chown root:root /mnt/etc/shadow
sudo chmod 640 /mnt/etc/shadow
sudo head -n 3 /mnt/etc/shadow
```

A malformed line denies access to every account on the system. The colon count is checked before the reboot.

**Unmount and boot the installed system**

```
sudo umount /mnt
sudo cryptsetup close crypt
sudo reboot
```

The `cryptsetup close` command applies only when LUKS was opened. The ISO is removed and the machine is accessed with the new credential.

**Failure conditions**

Incorrect partition selected: `cat /mnt/etc/os-release` identifies the mounted system.
Authentication fails for all accounts: a line in `/etc/shadow` is malformed, and the colon count is verified.
Authentication fails with correct syntax: enforcing SELinux blocks the credential. The volume is remounted and `sudo touch /mnt/.autorelabel` is executed before unmounting, after which the system relabels during startup.
