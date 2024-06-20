import subprocess


def run_command(command):
    try:
        # 运行命令并等待命令执行完成
        process = subprocess.run(command, shell=True, check=True)
        print(f"Command executed successfully: {command}")
    except subprocess.CalledProcessError:
        print(f"Failed to execute command: {command}")


def main():
    # 重采样为16000Hz
    run_command("python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000")

    # 重采样为32000Hz
    run_command("python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000")

    # 使用16K音频，提取音高
    run_command("python prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch")

    # 使用16K音频，提取内容编码（PPG）
    run_command("python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper")

    # 使用16K音频，提取内容编码（Hubert）
    run_command("python prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert")

    # 使用16K音频，提取音色编码
    run_command("python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker")

    # 使用32K音频，提取线性谱
    run_command("python prepare/preprocess_spec.py -w data_svc/waves-32k/ -s data_svc/specs")


if __name__ == "__main__":
    main()
