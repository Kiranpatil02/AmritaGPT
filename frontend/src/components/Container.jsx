import React, { useState } from "react";
import ChatInput from "./ChatInput";
import MessageContainer from "./MessageContainer";

const Container = () => {

  return (
    <div className="bg-[#DAD9D9] md:w-[70%] sm:w-[90%] mx-auto h-[80vh] rounded-2xl p-10 mt-2 relative overflow-y-hidden">
      <div className="w-full absolute bottom-6 left-0 right-0 mx-auto px-6">
        <div>
          <MessageContainer />
        </div>
        <div className="items-baseline">
          <ChatInput />
        </div>
      </div>
    </div>
  );
};

export default Container;