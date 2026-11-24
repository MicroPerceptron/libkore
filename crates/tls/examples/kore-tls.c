/* An offline Kore utility and a TLS client probe on hosts with TCP sockets. */
#define _POSIX_C_SOURCE 200809L
#include <kore/tls.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef KORE_TLS_OFFLINE
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#endif

static int fail(const char *message)
{
    fprintf(stderr, "tls: %s\n", message);
    return 1;
}

static void print_hex(const uint8_t *bytes, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        printf("%02x", (unsigned)bytes[i]);
    }
    puts("");
}

static int random_hex(const char *argument)
{
    char *end;
    unsigned long count = strtoul(argument, &end, 10);
    uint8_t bytes[1024];
    if (end == argument || *end || count == 0 || count > sizeof bytes) {
        return fail("rand expects a byte count from 1 to 1024");
    }
    if (kore_tls_random(bytes, count) != KORE_TLS_STATUS_OK) {
        return fail("secure entropy unavailable");
    }
    print_hex(bytes, count);
    return 0;
}

static int sha256_file(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return fail("could not open input file");
    }
    KoreTlsSha256 *hash = kore_tls_sha256_new();
    uint8_t bytes[4096];
    ssize_t n;
    while ((n = read(fd, bytes, sizeof bytes)) > 0) {
        kore_tls_sha256_update(hash, bytes, (size_t)n);
    }
    close(fd);
    if (n == 0) {
        uint8_t digest[32];
        kore_tls_sha256_digest(hash, digest);
        print_hex(digest, sizeof digest);
    }
    kore_tls_sha256_free(hash);
    return n == 0 ? 0 : fail("reading input file failed");
}

#ifndef KORE_TLS_OFFLINE
static int connect_tcp(const char *host, const char *port)
{
    struct addrinfo hints = {0}, *addresses;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &addresses)) {
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *p = addresses; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    return fd;
}

static int write_all(int fd, const uint8_t *bytes, size_t len)
{
    while (len != 0) {
        ssize_t n = write(fd, bytes, len);
        if (n <= 0) {
            return -1;
        }
        bytes += n;
        len -= (size_t)n;
    }
    return 0;
}

static int handshake(int fd, KoreTlsClient *client)
{
    uint8_t input[65536], output[KORE_TLS_OUTPUT_BYTES];
    size_t len = 0;
    uint32_t action = KORE_TLS_ACTION_POLL;
    for (;;) {
        KoreTlsProgress progress;
        KoreTlsStatus status = kore_tls_client_step(client, action, input, len,
                                                     NULL, 0, output, sizeof output, &progress);
        if (status == KORE_TLS_STATUS_CERTIFICATE) {
            return fail("peer certificate verification failed");
        }
        if (status != KORE_TLS_STATUS_OK) {
            return fail("TLS handshake failed");
        }
        len -= progress.consumed;
        memmove(input, input + progress.consumed, len);
        switch (progress.event) {
        case KORE_TLS_EVENT_CIPHERTEXT:
        case KORE_TLS_EVENT_SHUTDOWN_SENT:
            if (write_all(fd, output, progress.produced) != 0) {
                return fail("sending TLS records failed");
            }
            if (progress.event == KORE_TLS_EVENT_SHUTDOWN_SENT) {
                puts("TLS handshake and peer certificate verified");
                return 0;
            }
            break;
        case KORE_TLS_EVENT_READY:
            action = KORE_TLS_ACTION_SHUTDOWN;
            break;
        case KORE_TLS_EVENT_NEED_INPUT: {
            ssize_t n = read(fd, input + len, sizeof input - len);
            if (n <= 0) {
                return fail("peer stopped responding during the handshake");
            }
            len += (size_t)n;
            break;
        }
        default:
            return fail("unexpected data or close during the handshake");
        }
    }
}

static int verify_peer(const char *host, const char *port, const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return fail("could not open the PEM trust bundle");
    }
    uint8_t *roots = malloc(KORE_TLS_ROOTS_BYTES + 1);
    if (roots == NULL) {
        fclose(file);
        return fail("could not allocate a trust buffer");
    }
    size_t len = fread(roots, 1, KORE_TLS_ROOTS_BYTES + 1, file);
    int read_error = ferror(file);
    fclose(file);
    KoreTlsClient *client = NULL;
    KoreTlsStatus status = read_error ? KORE_TLS_STATUS_INVALID_TRUST :
        kore_tls_client_new((const uint8_t *)host, strlen(host), roots, len, &client);
    free(roots);
    if (status != KORE_TLS_STATUS_OK) {
        return fail("could not load the trust bundle or initialize TLS");
    }
    int fd = connect_tcp(host, port);
    if (fd < 0) {
        kore_tls_client_free(client);
        return fail("TCP connection failed");
    }
    const struct timeval timeout = {5, 0};
    int result;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout)
        || setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout)) {
        result = fail("could not configure socket timeouts");
    } else {
        result = handshake(fd, client);
    }
    close(fd);
    kore_tls_client_free(client);
    return result;
}
#endif

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "version") == 0) {
        puts("kore-tls (rustls)");
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "rand") == 0) {
        return random_hex(argv[2]);
    }
    if (argc == 4 && strcmp(argv[1], "dgst") == 0 && strcmp(argv[2], "-sha256") == 0) {
        return sha256_file(argv[3]);
    }
#ifndef KORE_TLS_OFFLINE
    if (argc == 5 && strcmp(argv[1], "s_client") == 0) {
        return verify_peer(argv[2], argv[3], argv[4]);
    }
    return fail("usage: version | rand BYTES | dgst -sha256 FILE | s_client HOST PORT CA.pem");
#else
    return fail("usage: version | rand BYTES | dgst -sha256 FILE");
#endif
}
