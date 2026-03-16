import java.security.SecureRandom;

public class Csprng{
	public static void main(String[] args) {
		SecureRandom sr = new SecureRandom();
		byte[] randomBytes  = new byte[32];
		sr.nextBytes(randomBytes);
		System.out.println(bytesToHex(randomBytes));
	}
	private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
