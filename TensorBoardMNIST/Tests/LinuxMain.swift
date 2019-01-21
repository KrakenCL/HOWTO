import XCTest

import TensorBoardMNISTTests

var tests = [XCTestCaseEntry]()
tests += TensorBoardMNISTTests.allTests()
XCTMain(tests)