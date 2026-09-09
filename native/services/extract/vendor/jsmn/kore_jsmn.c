#include "jsmn.h"
#include <stdint.h>

/*
 * The sole foreign boundary. No jsmn type crosses it: Rust supplies immutable
 * bytes and receives two integers after jsmn has validated the complete root.
 */
int32_t kore_jsmn_summary(const uint8_t *bytes, size_t len,
                          uint32_t *root_members, uint32_t *token_count) {
  jsmn_parser parser;
  jsmntok_t tokens[256];
  int parsed;

  if (bytes == NULL || root_members == NULL || token_count == NULL ||
      len == 0) {
    return JSMN_ERROR_INVAL;
  }
  jsmn_init(&parser);
  parsed = jsmn_parse(&parser, (const char *)bytes, len, tokens, 256);
  if (parsed < 1) {
    return parsed < 0 ? parsed : JSMN_ERROR_INVAL;
  }
  if (tokens[0].type != JSMN_OBJECT && tokens[0].type != JSMN_ARRAY) {
    return JSMN_ERROR_INVAL;
  }
  if (tokens[0].start < 0 || tokens[0].end < tokens[0].start) {
    return JSMN_ERROR_INVAL;
  }
  for (size_t i = (size_t)tokens[0].end; i < len; ++i) {
    if (bytes[i] != ' ' && bytes[i] != '\t' && bytes[i] != '\r' &&
        bytes[i] != '\n') {
      return JSMN_ERROR_INVAL;
    }
  }
  *root_members = (uint32_t)tokens[0].size;
  *token_count = (uint32_t)parsed;
  return 0;
}
