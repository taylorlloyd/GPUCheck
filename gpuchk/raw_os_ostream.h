#ifndef CUSTOM_RAW_OS_OSTREAM_H
#define CUSTOM_RAW_OS_OSTREAM_H

#include "llvm/Support/raw_ostream.h"
#include <iosfwd>

namespace llvm {

/// raw_os_ostream - A raw_ostream that writes to an std::ostream.  This is a
/// simple adaptor class.  It does not check for output errors; clients should
/// use the underlying stream to detect errors.
class raw_os_ostream : public raw_ostream {
  std::ostream &OS;

  /// write_impl - See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override {
    OS.write(Ptr, Size);
  }

  /// current_pos - Return the current position within the stream, not
  /// counting the bytes currently in the buffer.
  uint64_t current_pos() const override { return OS.tellp(); }

public:
  raw_os_ostream(std::ostream &O) : OS(O) {}
  ~raw_os_ostream() override {
    flush();
  }
};

} // end llvm namespace

#endif
