------------------------------------------------------------------------------
--                                                                          --
--                         GNAT RUN-TIME COMPONENTS                         --
--                                                                          --
--                         S Y S T E M . M E M O R Y                        --
--                                                                          --
--                                 B o d y                                  --
--                                                                          --
--          Copyright (C) 2001-2025, Free Software Foundation, Inc.         --
--                                                                          --
-- GNAT is free software;  you can  redistribute it  and/or modify it under --
-- terms of the  GNU General Public License as published  by the Free Soft- --
-- ware  Foundation;  either version 3,  or (at your option) any later ver- --
-- sion.  GNAT is distributed in the hope that it will be useful, but WITH- --
-- OUT ANY WARRANTY;  without even the  implied warranty of MERCHANTABILITY --
-- or FITNESS FOR A PARTICULAR PURPOSE.                                     --
--                                                                          --
-- As a special exception under Section 7 of GPL version 3, you are granted --
-- additional permissions described in the GCC Runtime Library Exception,   --
-- version 3.1, as published by the Free Software Foundation.               --
--                                                                          --
-- You should have received a copy of the GNU General Public License and    --
-- a copy of the GCC Runtime Library Exception along with this program;     --
-- see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see    --
-- <http://www.gnu.org/licenses/>.                                          --
--                                                                          --
-- GNAT was originally developed  by the GNAT team at  New York University. --
-- Extensive contributions were provided by Ada Core Technologies Inc.      --
--                                                                          --
------------------------------------------------------------------------------

--  This version contains allocation tracking capability

--  The object file corresponding to this instrumented version is to be found
--  in libgmem.

--  When enabled, the subsystem logs all the calls to __gnat_malloc and
--  __gnat_free. This log can then be processed by gnatmem to detect
--  dynamic memory leaks.

--  To use this functionality, you must compile your application with -g
--  and then link with this object file:

--     gnatmake -g program -largs -lgmem

--  After compilation, you may use your program as usual except that upon
--  completion, it will generate in the current directory the file gmem.out.

--  You can then investigate for possible memory leaks and mismatch by calling
--  gnatmem with this file as an input:

--    gnatmem -i gmem.out program

--  See gnatmem section in the GNAT User's Guide for more details

--  NOTE: This capability is currently supported on the following targets:

--    Windows
--    AIX
--    GNU/Linux
--    HP-UX
--    Solaris

--  NOTE FOR FUTURE PLATFORMS SUPPORT: It is assumed that type Duration is
--  64 bit. If the need arises to support architectures where this assumption
--  is incorrect, it will require changing the way timestamps of allocation
--  events are recorded.

pragma Source_File_Name (System.Memory, Body_File_Name => "memtrack.adb");

with Ada.Exceptions;
with GNAT.IO;

with System.Soft_Links;
with System.Traceback;
with System.Traceback_Entries;
with System.CRTL;
with System.OS_Lib;
with System.OS_Primitives;

package body System.Memory is

   use Ada.Exceptions;
   use System.Soft_Links;
   use System.Traceback;
   use System.Traceback_Entries;
   use GNAT.IO;

   function c_malloc (Size : size_t) return System.Address;
   pragma Import (C, c_malloc, "malloc");

   procedure c_free (Ptr : System.Address);
   pragma Import (C, c_free, "free");

   function c_realloc
     (Ptr : System.Address; Size : size_t) return System.Address;
   pragma Import (C, c_realloc, "realloc");

   In_Child_After_Fork : Integer;
   pragma Import (C, In_Child_After_Fork, "__gnat_in_child_after_fork");

   subtype File_Ptr is CRTL.FILEs;

   procedure Write (Ptr : System.Address; Size : size_t);

   procedure Putc (Char : Character);

   procedure Finalize;
   pragma Export (C, Finalize, "__gnat_finalize");
   --  Replace the default __gnat_finalize to properly close the log file

   Address_Size : constant := System.Address'Max_Size_In_Storage_Elements;
   --  Size in bytes of a pointer

   Max_Call_Stack : constant := 200;
   --  Maximum number of frames supported

   Tracebk   : Tracebacks_Array (1 .. Max_Call_Stack);
   Num_Calls : aliased Integer := 0;

   Gmemfname : constant String := "gmem.out" & ASCII.NUL;
   --  Allocation log of a program is saved in a file gmem.out
   --  ??? What about Ada.Command_Line.Command_Name & ".out" instead of static
   --  gmem.out

   Gmemfile : File_Ptr;
   --  Global C file pointer to the allocation log

   Needs_Init : Boolean := True;
   --  Reset after first call to Gmem_Initialize

   procedure Gmem_Initialize;
   --  Initialization routine; opens the file and writes a header string. This
   --  header string is used as a magic-tag to know if the .out file is to be
   --  handled by GDB or by the GMEM (instrumented malloc/free) implementation.

   First_Call : Boolean := True;
   --  Depending on implementation, some of the traceback routines may
   --  themselves do dynamic allocation. We use First_Call flag to avoid
   --  infinite recursion

   function Allow_Trace return Boolean;
   pragma Inline (Allow_Trace);
   --  Check if the memory trace is allowed

   -----------------
   -- Allow_Trace --
   -----------------

   function Allow_Trace return Boolean is
   begin
      if First_Call then
         First_Call := False;
         return In_Child_After_Fork = 0;
      else
         return False;
      end if;
   end Allow_Trace;

   -----------
   -- Alloc --
   -----------

   function Alloc (Size : size_t) return System.Address is
      Result      : aliased System.Address;
      Actual_Size : aliased size_t := Size;
      Timestamp   : aliased Duration;

   begin
      if Size = size_t'Last then
         Raise_Exception (Storage_Error'Identity, "object too large");
      end if;

      --  Change size from zero to non-zero. We still want a proper pointer
      --  for the zero case because pointers to zero length objects have to
      --  be distinct, but we can't just go ahead and allocate zero bytes,
      --  since some malloc's return zero for a zero argument.

      if Size = 0 then
         Actual_Size := 1;
      end if;

      Lock_Task.all;

      Result := c_malloc (Actual_Size);

      if Allow_Trace then

         --  Logs allocation call
         --  format is:
         --   'A' <mem addr> <size chunk> <len backtrace> <addr1> ... <addrn>

         if Needs_Init then
            Gmem_Initialize;
         end if;

         Timestamp := System.OS_Primitives.Clock;
         Call_Chain
           (Tracebk, Max_Call_Stack, Num_Calls, Skip_Frames => 2);
         Putc ('A');
         Write (Result'Address, Address_Size);
         Write (Actual_Size'Address, size_t'Max_Size_In_Storage_Elements);
         Write (Timestamp'Address, Duration'Max_Size_In_Storage_Elements);
         Write (Num_Calls'Address, Integer'Max_Size_In_Storage_Elements);

         for J in Tracebk'First .. Tracebk'First + Num_Calls - 1 loop
            declare
               Ptr : System.Address := PC_For (Tracebk (J));
            begin
               Write (Ptr'Address, Address_Size);
            end;
         end loop;

         First_Call := True;

      end if;

      Unlock_Task.all;

      if Result = System.Null_Address then
         Raise_Exception (Storage_Error'Identity, "heap exhausted");
      end if;

      return Result;
   end Alloc;

   --------------
   -- Finalize --
   --------------

   procedure Finalize is
   begin
      if not Needs_Init and then CRTL.fclose (Gmemfile) /= 0 then
         Put_Line ("gmem close error: " & OS_Lib.Errno_Message);
      end if;
   end Finalize;

   ----------
   -- Free --
   ----------

   procedure Free (Ptr : System.Address) is
      Addr      : aliased constant System.Address := Ptr;
      Timestamp : aliased Duration;

   begin
      Lock_Task.all;

      if Allow_Trace then

         --  Logs deallocation call
         --  format is:
         --   'D' <mem addr> <len backtrace> <addr1> ... <addrn>

         if Needs_Init then
            Gmem_Initialize;
         end if;

         Call_Chain
           (Tracebk, Max_Call_Stack, Num_Calls, Skip_Frames => 2);
         Timestamp := System.OS_Primitives.Clock;
         Putc ('D');
         Write (Addr'Address, Address_Size);
         Write (Timestamp'Address, Duration'Max_Size_In_Storage_Elements);
         Write (Num_Calls'Address, Integer'Max_Size_In_Storage_Elements);

         for J in Tracebk'First .. Tracebk'First + Num_Calls - 1 loop
            declare
               Ptr : System.Address := PC_For (Tracebk (J));
            begin
               Write (Ptr'Address, Address_Size);
            end;
         end loop;

         c_free (Ptr);

         First_Call := True;
      end if;

      Unlock_Task.all;
   end Free;

   ---------------------
   -- Gmem_Initialize --
   ---------------------

   procedure Gmem_Initialize is
      Timestamp : aliased Duration;
      File_Mode : constant String := "wb" & ASCII.NUL;
   begin
      if Needs_Init then
         Needs_Init := False;
         System.OS_Primitives.Initialize;
         Timestamp := System.OS_Primitives.Clock;
         Gmemfile := CRTL.fopen (Gmemfname'Address, File_Mode'Address);

         if Gmemfile = System.Null_Address then
            Put_Line ("Couldn't open gnatmem log file for writing");
            OS_Lib.OS_Exit (255);
         end if;

         declare
            S : constant String := "GMEM DUMP" & ASCII.LF;
         begin
            Write (S'Address, S'Length);
            Write (Timestamp'Address, Duration'Max_Size_In_Storage_Elements);
         end;
      end if;
   end Gmem_Initialize;

   ----------
   -- Putc --
   ----------

   procedure Putc (Char : Character) is
      C : constant Integer := Character'Pos (Char);

   begin
      if CRTL.fputc (C, Gmemfile) /= C then
         Put_Line ("gmem fputc error: " & OS_Lib.Errno_Message);
      end if;
   end Putc;

   -------------
   -- Realloc --
   -------------

   function Realloc
     (Ptr  : System.Address;
      Size : size_t) return System.Address
   is
      Addr      : aliased constant System.Address := Ptr;
      Result    : aliased System.Address;
      Timestamp : aliased Duration;

   begin
      --  For the purposes of allocations logging, we treat realloc as a free
      --  followed by malloc. This is not exactly accurate, but is a good way
      --  to fit it into malloc/free-centered reports.

      if Size = size_t'Last then
         Raise_Exception (Storage_Error'Identity, "object too large");
      end if;

      Abort_Defer.all;
      Lock_Task.all;

      if Allow_Trace then
         --  We first log deallocation call

         if Needs_Init then
            Gmem_Initialize;
         end if;
         Call_Chain
           (Tracebk, Max_Call_Stack, Num_Calls, Skip_Frames => 2);
         Timestamp := System.OS_Primitives.Clock;
         Putc ('D');
         Write (Addr'Address, Address_Size);
         Write (Timestamp'Address, Duration'Max_Size_In_Storage_Elements);
         Write (Num_Calls'Address, Integer'Max_Size_In_Storage_Elements);

         for J in Tracebk'First .. Tracebk'First + Num_Calls - 1 loop
            declare
               Ptr : System.Address := PC_For (Tracebk (J));
            begin
               Write (Ptr'Address, Address_Size);
            end;
         end loop;

         --  Now perform actual realloc

         Result := c_realloc (Ptr, Size);

         --   Log allocation call using the same backtrace

         Putc ('A');
         Write (Result'Address, Address_Size);
         Write (Size'Address, size_t'Max_Size_In_Storage_Elements);
         Write (Timestamp'Address, Duration'Max_Size_In_Storage_Elements);
         Write (Num_Calls'Address, Integer'Max_Size_In_Storage_Elements);

         for J in Tracebk'First .. Tracebk'First + Num_Calls - 1 loop
            declare
               Ptr : System.Address := PC_For (Tracebk (J));
            begin
               Write (Ptr'Address, Address_Size);
            end;
         end loop;

         First_Call := True;
      end if;

      Unlock_Task.all;
      Abort_Undefer.all;

      if Result = System.Null_Address then
         Raise_Exception (Storage_Error'Identity, "heap exhausted");
      end if;

      return Result;
   end Realloc;

   -----------
   -- Write --
   -----------

   procedure Write (Ptr : System.Address; Size : size_t) is
      function fwrite
        (buffer : System.Address;
         size   : size_t;
         count  : size_t;
         stream : File_Ptr) return size_t;
      pragma Import (C, fwrite);

   begin
      if fwrite (Ptr, Size, 1, Gmemfile) /= 1 then
         Put_Line ("gmem fwrite error: " & OS_Lib.Errno_Message);
      end if;
   end Write;

end System.Memory;
